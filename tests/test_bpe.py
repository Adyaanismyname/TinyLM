"""
Tests for BPE.py -- particularly the two bugs the rewrite targets:
process-dependent token IDs, and merges leaking across word boundaries.
"""

import os
import subprocess
import sys

import pytest

from BPE import BPETokenizer, count_words, merge_pair, split_words, train_bpe

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_TEXTS = [
    "Once upon a time there was a little cat named Tom.",
    "Tom the cat was very happy. Tom liked to play all day.",
    "One sunny day Tom met a dog. The dog was nice and friendly.",
    "Tom and the dog played together in the park until the sun went down.",
]


def build_tokenizer(vocab_size=80):
    word_counts = count_words(SAMPLE_TEXTS)
    base_chars, merges = train_bpe(word_counts, vocab_size=vocab_size, log_every=1000)
    return BPETokenizer(base_chars, merges)


def test_split_words_reconstructs_exactly():
    for text in SAMPLE_TEXTS + ["  leading and trailing spaces  ", "no_spaces_here", ""]:
        assert "".join(split_words(text)) == text


def test_merge_pair_basic():
    assert merge_pair(["a", "b", "c", "b", "c"], ("b", "c")) == ["a", "bc", "bc"]
    assert merge_pair(["a", "a", "a"], ("a", "a")) == ["aa", "a"]  # left-to-right, non-overlapping
    assert merge_pair(["x"], ("x", "y")) == ["x"]


def test_round_trip_encode_decode():
    tok = build_tokenizer()
    for text in SAMPLE_TEXTS:
        assert tok.decode(tok.encode(text)) == text


def test_vocab_size_respected():
    # train_bpe's vocab_size covers only the learned vocab (base chars +
    # merges); BPETokenizer adds special tokens on top, same convention as
    # dataset.py reserving a slot for <unk> before calling train_bpe.
    tok = build_tokenizer(vocab_size=80)
    assert len(tok.base_chars) + len(tok.merges) <= 80  # may stop early, never overshoots
    assert len(tok.vocab) == len(tok.base_chars) + len(tok.merges) + len(tok.special_tokens)


def test_ids_are_dense_and_unique():
    tok = build_tokenizer()
    ids = sorted(tok.token_to_id.values())
    assert ids == list(range(len(tok.vocab)))


def test_special_tokens_placed_last_and_deterministic_order():
    tok = build_tokenizer()
    assert tok.vocab[-2:] == ["<unk>", "<|endoftext|>"]
    assert tok.unk_token == "<unk>"
    assert tok.eos_token == "<|endoftext|>"
    assert tok.token_to_id["<unk>"] == len(tok.vocab) - 2
    assert tok.token_to_id["<|endoftext|>"] == len(tok.vocab) - 1


def test_unknown_char_falls_back_to_unk():
    tok = build_tokenizer()
    ids = tok.encode("☃")  # a snowman -- definitely not in this tiny corpus
    assert tok.unk_id in ids


def test_add_eos_appends_eos_id():
    tok = build_tokenizer()
    ids = tok.encode("Tom the cat.", add_eos=True)
    assert ids[-1] == tok.eos_id


def test_two_independent_instances_same_vocab():
    """The core determinism guarantee: build the tokenizer twice from the
    same learned merges and the token->id mapping must be identical."""
    word_counts = count_words(SAMPLE_TEXTS)
    base_chars, merges = train_bpe(word_counts, vocab_size=80, log_every=1000)

    tok_a = BPETokenizer(base_chars, merges)
    tok_b = BPETokenizer(base_chars, merges)
    assert tok_a.token_to_id == tok_b.token_to_id


def test_merges_never_cross_word_boundary():
    """A merge learned from "cat." should never fire across "cat" + " "
    (the space run) -- e.g. merging ('t', ' ') would corrupt every word
    boundary in the corpus."""
    word_counts = count_words(SAMPLE_TEXTS)
    _, merges = train_bpe(word_counts, vocab_size=80, log_every=1000)
    for (a, b), _ in merges:
        assert not (a.isspace() and not b.isspace()), f"merge {(a, b)} crosses into a word"
        assert not (b.isspace() and not a.isspace()), f"merge {(a, b)} crosses out of a word"


def test_save_and_load_roundtrip(tmp_path):
    tok = build_tokenizer()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    reloaded = BPETokenizer.load(path)

    assert reloaded.token_to_id == tok.token_to_id
    for text in SAMPLE_TEXTS:
        assert reloaded.encode(text) == tok.encode(text)
        assert reloaded.decode(reloaded.encode(text)) == text


def test_cross_process_determinism(tmp_path):
    """Reproduces the exact bug being fixed: run tokenizer construction in
    two fresh Python subprocesses (so PYTHONHASHSEED randomization, if the
    code depended on it, would actually differ between them) and check the
    resulting vocabularies match byte-for-byte."""
    script = tmp_path / "build_vocab.py"
    script.write_text(
        "import sys, json\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "from BPE import BPETokenizer, count_words, train_bpe\n"
        f"texts = {SAMPLE_TEXTS!r}\n"
        "word_counts = count_words(texts)\n"
        "base_chars, merges = train_bpe(word_counts, vocab_size=80, log_every=1000)\n"
        "tok = BPETokenizer(base_chars, merges)\n"
        "print(json.dumps(tok.token_to_id, sort_keys=True))\n",
        encoding="utf-8",
    )

    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": "random"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip().splitlines()[-1])

    assert outputs[0] == outputs[1]


def test_duplicate_token_raises():
    with pytest.raises(ValueError):
        BPETokenizer(["a", "b"], [(("a", "b"), "a")])  # "a" collides with a base char
