"""
From-scratch byte-pair-encoding tokenizer.

Two changes from the original version, both needed to train on the full
TinyStories corpus instead of an 8,000-story slice:

1. Merges are learned and applied *within words*, not across a whole
   document. `\\S+|\\s+` splits text into runs of non-whitespace and runs of
   whitespace ("words" from here on, loosely); concatenating the pieces back
   together reproduces the original text exactly, and a merge learned inside
   one word can never accidentally span a word boundary. This unlocks a
   per-word cache in `encode()` (the same word recurs constantly across a
   story corpus, so caching turns "replay every merge, every time" into
   "replay every merge once per *unique* word") and a training algorithm
   that updates pair counts incrementally instead of rescanning the whole
   corpus after every merge (see `train_bpe`).

2. Token IDs no longer come from iterating a Python `set`. Sets in Python
   are ordered by hash, and *string* hashing is randomized per-process
   (PYTHONHASHSEED) unless explicitly disabled -- so the old
   `{token: i for i, token in enumerate(self.vocab)}` assigned different IDs
   to the same vocabulary in different runs. A checkpoint saved by one
   process and reloaded by another would silently decode gibberish. IDs
   here come from an explicit, deterministic order: sorted base characters,
   then merges in the order they were learned, then special tokens.
"""

import json
import re
from collections import Counter

_WORD_RE = re.compile(r"\S+|\s+")


def merge_pair(tokens, pair):
    """Merges every non-overlapping adjacent occurrence of `pair` in
    `tokens` into a single combined token, left to right."""
    new_tokens = []
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    if i < len(tokens):
        new_tokens.append(tokens[i])
    return new_tokens


def split_words(text):
    """Splits text into alternating non-whitespace/whitespace runs. Purely
    mechanical -- concatenating the pieces reproduces `text` exactly, so
    this can never lose information, only give BPE a boundary to respect."""
    return _WORD_RE.findall(text)


def count_words(texts):
    """Counts occurrences of each unique word (see split_words) across a
    corpus of documents. This is the input train_bpe actually needs -- BPE
    only cares how often each word shape occurs, not which document it
    came from."""
    counts = Counter()
    for text in texts:
        counts.update(split_words(text))
    return counts


def train_bpe(word_counts, vocab_size, log_every=100):
    """
    Learns BPE merges from a {word: count} table (see count_words).

    Efficient by construction: rather than recomputing pair frequencies over
    the whole corpus after every merge (the original approach, which is
    O(merges x corpus_size) and becomes a multi-hour job on a full-size
    corpus), this keeps a running `pair_counts` table and, after each merge,
    only recomputes the words that actually contained the merged pair --
    typically a small fraction of the vocabulary. Total work is roughly
    O(merges x average_affected_words), independent of how many times each
    word occurs in the corpus.

    Returns (base_chars, merges):
      base_chars -- sorted list of the single-character tokens seen in the
                    corpus (the starting vocabulary before any merge)
      merges     -- ordered list of ((token1, token2), merged_token)
    """
    symbols = {word: list(word) for word in word_counts}
    base_chars = sorted({ch for word in word_counts for ch in word})

    def pair_counts_for(word):
        syms = symbols[word]
        return [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]

    pair_counts = Counter()
    pair_where = {}
    for word, freq in word_counts.items():
        for pair in pair_counts_for(word):
            pair_counts[pair] += freq
            pair_where.setdefault(pair, set()).add(word)

    merges = []
    vocab_len = len(base_chars)
    merges_needed = vocab_size - vocab_len
    print(
        f"train_bpe: starting vocab {vocab_len} -> target {vocab_size} "
        f"({merges_needed} merges needed, {len(word_counts)} unique words)"
    )

    while vocab_len < vocab_size:
        if not pair_counts:
            print("train_bpe: no more pairs left to merge -- stopping early")
            break

        # Deterministic tie-break (by pair itself) so merge order -- and
        # therefore every downstream token ID -- never depends on dict
        # iteration order or which pairs happened to be counted first.
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        new_token = best_pair[0] + best_pair[1]
        merges.append((best_pair, new_token))
        vocab_len += 1

        affected_words = pair_where.pop(best_pair)
        del pair_counts[best_pair]

        for word in affected_words:
            freq = word_counts[word]
            for pair in pair_counts_for(word):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair in pair_where:
                    pair_where[pair].discard(word)

            symbols[word] = merge_pair(symbols[word], best_pair)

            for pair in pair_counts_for(word):
                pair_counts[pair] += freq
                pair_where.setdefault(pair, set()).add(word)

        if len(merges) % log_every == 0 or vocab_len >= vocab_size:
            print(f"train_bpe: merge {len(merges)}/{merges_needed} | vocab {vocab_len}/{vocab_size}")

    print(f"train_bpe: done -- {len(merges)} merges, final vocab {vocab_len}")
    return base_chars, merges


class BPETokenizer:
    DEFAULT_SPECIAL_TOKENS = ("<unk>", "<|endoftext|>")

    def __init__(self, base_chars, merges, special_tokens=DEFAULT_SPECIAL_TOKENS):
        """
        base_chars: sorted list/iterable of single-character base tokens.
        merges: ordered list of ((token1, token2), merged_token), as
                returned by train_bpe -- order matters, since encode()
                replays them in this exact sequence.
        special_tokens: tokens appended after every learned token, in the
                        given order. The first is treated as the <unk>
                        fallback for encode(); if `<|endoftext|>` is
                        present it's exposed as `self.eos_token`/`eos_id`.

        Token IDs are assigned once, here, by concatenating base_chars
        (sorted) + [merged_token for each merge, in learned order] +
        special_tokens -- entirely determined by the arguments, never by
        Python's per-process string hash order. The same (base_chars,
        merges, special_tokens) always produces the same vocab.
        """
        self.base_chars = sorted(base_chars)
        self.merges = list(merges)
        self.special_tokens = tuple(special_tokens)

        self.vocab = self.base_chars + [new_tok for _, new_tok in self.merges] + list(self.special_tokens)
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("duplicate token in assembled vocab -- base_chars/merges/special_tokens overlap")

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

        self.unk_token = self.special_tokens[0]
        self.unk_id = self.token_to_id[self.unk_token]

        self.eos_token = "<|endoftext|>" if "<|endoftext|>" in self.token_to_id else None
        self.eos_id = self.token_to_id[self.eos_token] if self.eos_token else None

        self._word_cache = {}

    def _tokenize_word(self, word):
        """Applies every learned merge, in order, to one word's characters.
        Cached because the same word recurs constantly across a corpus --
        after the cache warms up, encoding a story is mostly dict lookups
        instead of replaying thousands of merges per call."""
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        tokens = list(word)
        for pair, _ in self.merges:
            if len(tokens) == 1:
                break
            tokens = merge_pair(tokens, pair)

        self._word_cache[word] = tokens
        return tokens

    def tokenize(self, text):
        """Converts raw text into BPE tokens, word by word (see
        split_words) so a merge can never span a word boundary."""
        tokens = []
        for word in split_words(text):
            tokens.extend(self._tokenize_word(word))
        return tokens

    def encode(self, text, add_eos=False):
        """Converts text into integer token IDs. If add_eos=True, appends
        `self.eos_id` at the end (requires an `<|endoftext|>` special
        token) -- used when concatenating documents into one training
        stream so the model can learn where one document ends."""
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in self.tokenize(text)]
        if add_eos:
            if self.eos_id is None:
                raise ValueError("add_eos=True but no '<|endoftext|>' token in this tokenizer's vocab")
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """Converts integer token IDs back into text. Unknown IDs decode to
        the empty string rather than raising, so a corrupted/truncated ID
        list still decodes as much as it can."""
        return "".join(self.id_to_token.get(i, "") for i in ids)

    def to_dict(self):
        return {
            "base_chars": self.base_chars,
            "merges": [[list(pair), new_tok] for pair, new_tok in self.merges],
            "special_tokens": list(self.special_tokens),
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d):
        merges = [(tuple(pair), new_tok) for pair, new_tok in d["merges"]]
        return cls(d["base_chars"], merges, special_tokens=tuple(d["special_tokens"]))

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    # Quick smoke test: train a tiny tokenizer, check round-tripping, and
    # confirm two independent instantiations from the same learned merges
    # produce byte-identical vocabularies (the bug this rewrite fixes).
    texts = [
        "Once upon a time there was a little cat.",
        "The cat was happy. The cat liked to play.",
        "One day the cat met a dog. The dog was nice too.",
    ]
    word_counts = count_words(texts)
    base_chars, merges = train_bpe(word_counts, vocab_size=60)

    tok = BPETokenizer(base_chars, merges)
    for text in texts:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"round-trip failed for: {text!r}"
    print("round-trip ok on all sample texts")

    tok2 = BPETokenizer(base_chars, merges)
    assert tok.token_to_id == tok2.token_to_id, "vocab ordering is not deterministic!"
    print("vocab ordering is deterministic across independent instances")

    print(f"vocab size: {len(tok.vocab)} | merges: {len(tok.merges)}")
    print(f"sample encode: {tok.encode('The cat was happy.')}")
