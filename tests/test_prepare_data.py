"""
End-to-end test of prepare_data.py's pipeline at small scale: real network
access (cached HF datasets), but tiny subsample sizes so it runs in seconds,
not minutes. Verifies the .bin files it writes are internally consistent and
loadable by data.py -- the actual contract every experiment script depends on.
"""

import json
import os

import numpy as np
import pytest

from BPE import BPETokenizer
from data import stream_from_bin
from prepare_data import run


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("data"))
    meta = run(
        out_dir=out_dir,
        vocab_size=120,
        tokenizer_train_stories=200,
        tinystories_num_dev=20,
        instruct_train_examples=200,
        instruct_dev_examples=20,
        instruct_test_examples=50,
        seed=0,
        tinystories_train_limit=1000,
        tinystories_test_limit=200,
    )
    return out_dir, meta


def test_meta_json_matches_returned_meta(prepared):
    out_dir, meta = prepared
    with open(os.path.join(out_dir, "meta.json")) as f:
        on_disk = json.load(f)
    assert on_disk == meta


def test_tokenizer_saved_and_loadable(prepared):
    out_dir, meta = prepared
    tok = BPETokenizer.load(os.path.join(out_dir, "tokenizer.json"))
    assert len(tok.vocab) <= 120
    assert tok.eos_token == "<|endoftext|>"
    assert tok.unk_token == "<unk>"


def test_bin_file_sizes_match_reported_token_counts(prepared):
    out_dir, meta = prepared
    for name, key in [
        ("tinystories_train.bin", "tinystories_train_tokens"),
        ("tinystories_dev.bin", "tinystories_dev_tokens"),
        ("tinystories_test.bin", "tinystories_test_tokens"),
        ("instruct_train.bin", "instruct_train_tokens"),
        ("instruct_dev.bin", "instruct_dev_tokens"),
        ("instruct_test.bin", "instruct_test_tokens"),
    ]:
        path = os.path.join(out_dir, name)
        arr = np.fromfile(path, dtype=np.uint16)
        assert len(arr) == meta[key], f"{name}: file has {len(arr)} tokens, meta says {meta[key]}"


def test_instruct_mask_files_same_length_as_token_files(prepared):
    out_dir, _ = prepared
    for split in ["train", "dev", "test"]:
        ids = np.fromfile(os.path.join(out_dir, f"instruct_{split}.bin"), dtype=np.uint16)
        mask = np.fromfile(os.path.join(out_dir, f"instruct_{split}_mask.bin"), dtype=np.uint8)
        assert len(ids) == len(mask)
        assert set(np.unique(mask).tolist()) <= {0, 1}


def test_instruct_mask_has_both_header_and_story_tokens(prepared):
    """Sanity check the mask isn't degenerate (all-0 or all-1) -- both
    masked header tokens and unmasked story tokens should be present."""
    out_dir, _ = prepared
    mask = np.fromfile(os.path.join(out_dir, "instruct_train_mask.bin"), dtype=np.uint8)
    assert mask.min() == 0
    assert mask.max() == 1
    assert 0 < mask.mean() < 1


def test_tinystories_bins_are_eos_separated(prepared):
    out_dir, meta = prepared
    tok = BPETokenizer.load(os.path.join(out_dir, "tokenizer.json"))
    arr = np.fromfile(os.path.join(out_dir, "tinystories_train.bin"), dtype=np.uint16)
    eos_count = int((arr == tok.eos_id).sum())
    assert eos_count == meta["tinystories_train_stories"]


def test_streams_loadable_by_data_module(prepared):
    out_dir, _ = prepared
    stream = stream_from_bin(os.path.join(out_dir, "tinystories_train.bin"), block_size=32)
    assert len(stream) > 0

    masked_stream = stream_from_bin(
        os.path.join(out_dir, "instruct_train.bin"), block_size=32,
        mask_path=os.path.join(out_dir, "instruct_train_mask.bin"),
    )
    assert len(masked_stream) > 0
    x, y, m = masked_stream.window(0)
    assert m is not None
    assert m.shape == y.shape


def test_words_eval_prompts_reference_real_words(prepared):
    out_dir, meta = prepared
    with open(os.path.join(out_dir, "words_eval_prompts.json")) as f:
        prompts = json.load(f)
    assert len(prompts) == meta["words_eval_prompts"]
    assert len(prompts) > 0
    for header, words in prompts:
        assert header.endswith("Story:")
        assert len(words) > 0


def test_no_story_overlap_between_tinystories_train_and_dev(prepared):
    """split_dev's no-overlap guarantee, exercised through the real
    pipeline: dev stories should not also appear (verbatim) in train."""
    from dataset import load_tinystories_texts, split_dev

    train_texts = load_tinystories_texts("train", num_examples=1000)
    train_pool, dev_texts = split_dev(train_texts, num_dev=20, seed=0)
    assert set(dev_texts).isdisjoint(train_pool)
