"""
Downloads TinyStories + TinyStoriesInstruct, trains the BPE tokenizer, and
writes everything every experiment script needs as uint16 memmap .bin files
plus a JSON tokenizer -- the one-time pipeline that replaces re-tokenizing
an 8,000-story slice inside every experiment script.

Run once before any experiment:

    python3 prepare_data.py

Writes, all under --out-dir (default "data/"):
  tokenizer.json                          -- the trained BPETokenizer
  tinystories_train.bin                   -- pretraining stream (EOS-separated stories)
  tinystories_dev.bin                     -- held out from train, for picking LRs
  tinystories_test.bin                    -- the official TinyStories validation split
  instruct_train.bin / instruct_train_mask.bin   -- fine-tuning stream + loss mask
  instruct_dev.bin   / instruct_dev_mask.bin     -- held out from Instruct train
  instruct_test.bin  / instruct_test_mask.bin    -- the official Instruct validation split
  words_eval_prompts.json                 -- held-out (header, target_words) pairs for
                                              Experiment 3's Words-constraint-rate metric
  meta.json                               -- token/example counts and the config used,
                                              so a report can cite exactly what these
                                              files contain without re-deriving it

Subsample sizes below are deliberately smaller than "the entire corpus":
BPE only needs a representative sample of word frequencies to learn a
stable vocabulary (see BPE.py), and Experiment 3's fine-tuning arms only
need a token budget on the order of ~20M tokens each (see the paper's
experiment plan) -- pulling in TinyStoriesInstruct's full ~21.8M rows would
cost a lot of encoding time for data that would mostly never get sampled.
Pretraining (tinystories_train.bin) does use the *entire* TinyStories train
split, since "does more data help" (Experiment 1) is one of the questions
being asked.
"""

import argparse
import json
import os
import time

import numpy as np

from BPE import BPETokenizer, count_words, train_bpe
from dataset import (
    format_instruct_prompt,
    load_tinystories_instruct_examples,
    load_tinystories_texts,
    split_dev,
)

VOCAB_SIZE = 1024
TOKENIZER_TRAIN_STORIES = 200_000   # subsample for learning BPE merges
TINYSTORIES_NUM_DEV = 10_000        # held out from train for LR selection (Experiment 2)

INSTRUCT_TRAIN_EXAMPLES = 300_000   # subsample -- see module docstring
INSTRUCT_DEV_EXAMPLES = 5_000
INSTRUCT_TEST_EXAMPLES = 5_000      # subsample of the official validation split
WORDS_EVAL_NUM_PROMPTS = 500        # for Experiment 3's Words-constraint-rate metric

SPLIT_SEED = 0


def log(msg):
    print(f"[prepare_data] {msg}", flush=True)


def write_token_stream(path, texts, tokenizer, progress_every=200_000):
    """Encodes every text in `texts` (each EOS-terminated) and writes the
    concatenated result as a uint16 memmap at `path`. Returns the total
    token count. Writes incrementally to a growable list-of-arrays rather
    than one giant Python list of ints, since the full TinyStories train
    split encodes to on the order of hundreds of millions of tokens."""
    chunks = []
    total = 0
    start = time.time()
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_eos=True)
        chunks.append(np.array(ids, dtype=np.uint16))
        total += len(ids)
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            log(f"  encoded {i + 1:,}/{len(texts):,} texts | {total:,} tokens so far "
                f"| {elapsed:.1f}s elapsed | ~{rate:.0f} texts/s")

    arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr.tofile(path)
    log(f"wrote {path} -- {total:,} tokens ({arr.nbytes / 1e6:.1f} MB)")
    return total


def encode_instruct_example(tokenizer, example):
    """Encodes one parsed Instruct example (dataset.py's dict format) into
    (ids, mask): header tokens get mask 0 (excluded from loss), story
    tokens (plus the trailing EOS) get mask 1. Encoding header and story as
    separate calls and concatenating the id lists gives the exact same
    tokens as encoding the joined string would -- format_instruct_prompt's
    header/story split always falls on a whitespace-run boundary (the
    header ends in "Story:" with no trailing space; the story starts with
    a leading space), and BPE.py's word-level tokenization never merges
    across such a boundary (see BPE.py's own test of that property)."""
    header, story = format_instruct_prompt(example)
    header_ids = tokenizer.encode(header)
    story_ids = tokenizer.encode(story, add_eos=True)
    ids = header_ids + story_ids
    mask = [0] * len(header_ids) + [1] * len(story_ids)
    return ids, mask


def write_instruct_stream(path_ids, path_mask, examples, tokenizer, progress_every=50_000):
    """Same idea as write_token_stream, but also writes the parallel
    uint8 loss-mask memmap data.stream_from_bin expects."""
    id_chunks, mask_chunks = [], []
    total = 0
    start = time.time()
    for i, example in enumerate(examples):
        ids, mask = encode_instruct_example(tokenizer, example)
        id_chunks.append(np.array(ids, dtype=np.uint16))
        mask_chunks.append(np.array(mask, dtype=np.uint8))
        total += len(ids)
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            log(f"  encoded {i + 1:,}/{len(examples):,} examples | {total:,} tokens so far "
                f"| {elapsed:.1f}s elapsed")

    ids_arr = np.concatenate(id_chunks) if id_chunks else np.array([], dtype=np.uint16)
    mask_arr = np.concatenate(mask_chunks) if mask_chunks else np.array([], dtype=np.uint8)
    assert len(ids_arr) == len(mask_arr)

    for path, arr in [(path_ids, ids_arr), (path_mask, mask_arr)]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arr.tofile(path)
    log(f"wrote {path_ids} + {path_mask} -- {total:,} tokens, "
        f"{int(mask_arr.sum()):,} loss-counted ({100 * mask_arr.mean():.1f}%)")
    return total


def build_words_eval_prompts(examples, num_prompts, seed=0):
    """Selects up to `num_prompts` Instruct examples that actually have a
    Words: field (some don't -- see dataset.py) and returns
    [(header_text, target_words), ...] for metrics.words_constraint_rate."""
    with_words = [ex for ex in examples if ex["words"]]
    _, sample = split_dev(with_words, num_dev=min(num_prompts, len(with_words)), seed=seed)
    return [(format_instruct_prompt(ex)[0], ex["words"]) for ex in sample]


def run(out_dir="data", tokenizer_train_stories=TOKENIZER_TRAIN_STORIES,
        tinystories_num_dev=TINYSTORIES_NUM_DEV, instruct_train_examples=INSTRUCT_TRAIN_EXAMPLES,
        instruct_dev_examples=INSTRUCT_DEV_EXAMPLES, instruct_test_examples=INSTRUCT_TEST_EXAMPLES,
        vocab_size=VOCAB_SIZE, seed=SPLIT_SEED,
        tinystories_train_limit=None, tinystories_test_limit=None):
    """
    tinystories_train_limit/tinystories_test_limit: cap how many TinyStories
    train/validation stories are loaded in the first place, instead of the
    full ~2.12M/~22K. None (the default) loads everything, which is what
    Experiment 1 ("does more data help") needs -- set these only for fast
    small-scale runs (tests, smoke checks), never for a real experiment.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = {"config": {
        "vocab_size": vocab_size, "tokenizer_train_stories": tokenizer_train_stories,
        "tinystories_num_dev": tinystories_num_dev, "instruct_train_examples": instruct_train_examples,
        "instruct_dev_examples": instruct_dev_examples, "instruct_test_examples": instruct_test_examples,
        "seed": seed,
    }}

    # ---- TinyStories: load, split, train tokenizer ----------------------
    log("loading TinyStories (train + validation)...")
    train_texts = load_tinystories_texts("train", num_examples=tinystories_train_limit)
    test_texts = load_tinystories_texts("validation", num_examples=tinystories_test_limit)
    log(f"loaded {len(train_texts):,} train / {len(test_texts):,} validation stories")

    train_pool, dev_texts = split_dev(train_texts, num_dev=tinystories_num_dev, seed=seed)
    log(f"held out {len(dev_texts):,} dev stories from train; {len(train_pool):,} remain for training")

    log(f"training BPE tokenizer on {min(tokenizer_train_stories, len(train_pool)):,} stories "
        f"(target vocab {vocab_size})...")
    tokenizer_texts = train_pool[:tokenizer_train_stories]
    word_counts = count_words(tokenizer_texts)
    # Reserve 2 slots for special tokens (<unk>, <|endoftext|>), added by
    # BPETokenizer's constructor on top of whatever train_bpe learns.
    base_chars, merges = train_bpe(word_counts, vocab_size=vocab_size - 2)
    tokenizer = BPETokenizer(base_chars, merges)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    log(f"tokenizer: {len(tokenizer.vocab)} tokens ({len(tokenizer.base_chars)} base chars, "
        f"{len(tokenizer.merges)} merges)")

    # ---- TinyStories: encode every split ---------------------------------
    log(f"encoding {len(train_pool):,} TinyStories train stories (this is the slow step)...")
    meta["tinystories_train_tokens"] = write_token_stream(
        os.path.join(out_dir, "tinystories_train.bin"), train_pool, tokenizer
    )
    meta["tinystories_dev_tokens"] = write_token_stream(
        os.path.join(out_dir, "tinystories_dev.bin"), dev_texts, tokenizer
    )
    meta["tinystories_test_tokens"] = write_token_stream(
        os.path.join(out_dir, "tinystories_test.bin"), test_texts, tokenizer
    )
    meta["tinystories_train_stories"] = len(train_pool)
    meta["tinystories_dev_stories"] = len(dev_texts)
    meta["tinystories_test_stories"] = len(test_texts)

    # ---- TinyStoriesInstruct: load, split, encode with loss mask ---------
    log(f"loading {instruct_train_examples:,} TinyStoriesInstruct train examples...")
    instruct_pool = load_tinystories_instruct_examples("train", num_examples=instruct_train_examples)
    instruct_train, instruct_dev = split_dev(instruct_pool, num_dev=min(instruct_dev_examples, len(instruct_pool) - 1), seed=seed)
    log(f"{len(instruct_train):,} instruct train / {len(instruct_dev):,} instruct dev examples")

    log(f"loading {instruct_test_examples:,} TinyStoriesInstruct validation examples...")
    instruct_test = load_tinystories_instruct_examples("validation", num_examples=instruct_test_examples)

    log("encoding instruct train split...")
    meta["instruct_train_tokens"] = write_instruct_stream(
        os.path.join(out_dir, "instruct_train.bin"), os.path.join(out_dir, "instruct_train_mask.bin"),
        instruct_train, tokenizer,
    )
    log("encoding instruct dev split...")
    meta["instruct_dev_tokens"] = write_instruct_stream(
        os.path.join(out_dir, "instruct_dev.bin"), os.path.join(out_dir, "instruct_dev_mask.bin"),
        instruct_dev, tokenizer,
    )
    log("encoding instruct test split...")
    meta["instruct_test_tokens"] = write_instruct_stream(
        os.path.join(out_dir, "instruct_test.bin"), os.path.join(out_dir, "instruct_test_mask.bin"),
        instruct_test, tokenizer,
    )
    meta["instruct_train_examples"] = len(instruct_train)
    meta["instruct_dev_examples"] = len(instruct_dev)
    meta["instruct_test_examples"] = len(instruct_test)

    # ---- Experiment 3's Words-constraint-rate held-out prompt set --------
    words_prompts = build_words_eval_prompts(instruct_test, WORDS_EVAL_NUM_PROMPTS, seed=seed)
    with open(os.path.join(out_dir, "words_eval_prompts.json"), "w") as f:
        json.dump(words_prompts, f, indent=2)
    meta["words_eval_prompts"] = len(words_prompts)
    log(f"wrote words_eval_prompts.json -- {len(words_prompts)} prompts")

    meta["eos_id"] = tokenizer.eos_id
    meta["unk_id"] = tokenizer.unk_id
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote meta.json")
    log("done.")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--tokenizer-train-stories", type=int, default=TOKENIZER_TRAIN_STORIES)
    parser.add_argument("--tinystories-num-dev", type=int, default=TINYSTORIES_NUM_DEV)
    parser.add_argument("--instruct-train-examples", type=int, default=INSTRUCT_TRAIN_EXAMPLES)
    parser.add_argument("--instruct-dev-examples", type=int, default=INSTRUCT_DEV_EXAMPLES)
    parser.add_argument("--instruct-test-examples", type=int, default=INSTRUCT_TEST_EXAMPLES)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--tinystories-train-limit", type=int, default=None,
                         help="Cap TinyStories train stories loaded (default: all ~2.12M). Testing only.")
    parser.add_argument("--tinystories-test-limit", type=int, default=None,
                         help="Cap TinyStories validation stories loaded (default: all ~22K). Testing only.")
    args = parser.parse_args()

    run(
        out_dir=args.out_dir, vocab_size=args.vocab_size,
        tokenizer_train_stories=args.tokenizer_train_stories,
        tinystories_num_dev=args.tinystories_num_dev,
        instruct_train_examples=args.instruct_train_examples,
        instruct_dev_examples=args.instruct_dev_examples,
        instruct_test_examples=args.instruct_test_examples,
        seed=args.seed,
        tinystories_train_limit=args.tinystories_train_limit,
        tinystories_test_limit=args.tinystories_test_limit,
    )
