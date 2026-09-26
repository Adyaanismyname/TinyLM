"""
Evaluation metrics for models trained by train.py / run_experiments.py.

Loads checkpoints/checkpoint.pt directly (model weights + the exact
tokenizer it was trained with, saved via BPETokenizer.to_dict()) rather than
rebuilding the dataset pipeline, so the tokenizer here always matches the
checkpointed model instead of accidentally retraining a new BPE vocab.

Computes:
  - perplexity / cross-entropy loss, evaluated over every window of the
    validation set (train.py's estimate_loss only samples random windows,
    which is fine for a training-time progress readout but not precise
    enough for a final benchmark number); optionally loss-masked, for
    instruction-tuning data where only "Story:" tokens should count
  - bits-per-character, which normalizes loss by character count instead
    of token count -- useful because it stays comparable even if you change
    vocab_size or retrain the tokenizer, unlike perplexity
  - a unigram frequency baseline, so you can tell whether the model has
    actually learned something beyond raw token frequency
  - top-1 / top-5 next-token accuracy
  - generation diversity (distinct-n), to catch degenerate/repetitive output
  - the Experiment 3 "Words-constraint rate": whether a fine-tuned model's
    generated story actually uses the words it was asked to include
  - inference throughput (tokens/sec) and generation latency, including
    percentile summaries (median/IQR) for the noisier per-step timings
    Experiments 4 and 5 rely on
  - peak memory, per backend (CUDA/MPS/CPU)

Run standalone:

    python3 metrics.py
"""

import math
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from BPE import BPETokenizer
from dataset import preprocess_text
from model import GPT, IGNORE_INDEX
from train import get_device, sync_device


def load_checkpoint(path="checkpoints/checkpoint.pt"):
    device = get_device()
    checkpoint = torch.load(path, map_location=device)

    tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])

    model = GPT(vocab_size=len(tokenizer.vocab), **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device


def load_split_token_ids(tokenizer, split, num_examples):
    """
    Loads a TinyStories split and encodes it with an already-trained
    tokenizer (no BPE retraining), so results line up with the checkpointed
    model. Returns (token_id_lists, raw_texts) -- the raw texts are needed
    for bits-per-character. Each story is EOS-terminated, matching how
    prepare_data.py builds the training stream.
    """
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories")
    texts = [preprocess_text(t) for t in dataset[split]["text"][:num_examples]]
    token_ids = [tokenizer.encode(t, add_eos=True) for t in texts]
    return token_ids, texts


def flatten(token_id_lists):
    flat = [tid for story in token_id_lists for tid in story]
    return torch.tensor(flat, dtype=torch.long)


def _stack_slices(data, idxs, start_offset, block_size):
    """Builds a (len(idxs), block_size) int64 tensor from slices of `data`,
    which may be a torch tensor (train.py's legacy flatten() output) or a
    numpy array/memmap (data.py's TokenStream.data). Routing both through
    np.asarray first (a no-op/view for either CPU tensors or numpy arrays)
    keeps _windows agnostic to which one it was handed."""
    rows = [np.asarray(data[i * block_size + start_offset : i * block_size + start_offset + block_size]) for i in idxs]
    return torch.from_numpy(np.stack(rows).astype(np.int64))


def _windows(data, block_size, batch_size, mask=None):
    """Yields (x, y[, mask_y]) batches covering every non-overlapping window
    in data exactly once, in order (no shuffling -- exhaustive evaluation
    doesn't need it, only full coverage). `mask`, if given, is a same-length
    0/1 array; mask_y is its slice aligned to y (mask[i] gates whether
    target token i counts), matching data.py's TokenStream convention."""
    num_windows = (len(data) - 1) // block_size
    for start in range(0, num_windows, batch_size):
        idxs = range(start, min(start + batch_size, num_windows))
        x = _stack_slices(data, idxs, 0, block_size)
        y = _stack_slices(data, idxs, 1, block_size)
        if mask is None:
            yield x, y, None
        else:
            m = _stack_slices(mask, idxs, 1, block_size)
            yield x, y, m


@torch.no_grad()
def perplexity(model, data, block_size, batch_size, device, mask=None):
    """Exact (non-sampled) average cross-entropy loss and perplexity over
    all of `data`. With `mask`, positions where mask == 0 are excluded from
    both the loss and the token count (e.g. instruction-tuning header
    tokens) -- so the reported perplexity is "perplexity over the tokens
    that were actually supposed to be predicted."""
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for x, y, m in _windows(data, block_size, batch_size, mask=mask):
        x, y = x.to(device), y.to(device)
        if m is not None:
            y = y.clone()
            y[m.to(device) == 0] = IGNORE_INDEX
            n = int((m != 0).sum().item())
        else:
            n = x.numel()
        if n == 0:
            continue
        _, loss, _ = model(x, y)
        total_loss += loss.item() * n
        total_tokens += n

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def bits_per_char(avg_loss_nats, token_ids, texts):
    """Cross-entropy re-expressed as bits per character of the original text."""
    total_tokens = sum(len(ids) for ids in token_ids)
    total_chars = sum(len(t) for t in texts)
    nats_per_char = avg_loss_nats * total_tokens / total_chars
    return nats_per_char / math.log(2)


def unigram_baseline_perplexity(token_ids, vocab_size):
    """
    Perplexity of the "always predict by training-set token frequency"
    baseline -- the floor any trained model should clear. If your model's
    perplexity is close to this, it hasn't learned much beyond unigram stats.
    """
    counts = Counter(tid for ids in token_ids for tid in ids)
    total = sum(counts.values())
    # Laplace smoothing so a token missing from `counts` wouldn't cause log(0)
    # (not reachable here since counts is built from the same token_ids, but
    # keeps the formula correct if reused with a different reference corpus).
    avg_nll = -sum(
        c * math.log((c + 1) / (total + vocab_size))
        for c in counts.values()
    ) / total
    return math.exp(avg_nll)


@torch.no_grad()
def topk_accuracy(model, data, block_size, batch_size, device, ks=(1, 5)):
    """Fraction of positions where the true next token lands in the model's top-k predictions."""
    model.eval()
    correct = {k: 0 for k in ks}
    total = 0
    max_k = max(ks)

    for x, y, _ in _windows(data, block_size, batch_size):
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        topk_ids = logits.topk(max_k, dim=-1).indices  # (batch, seq_len, max_k)

        for k in ks:
            hit = (topk_ids[:, :, :k] == y.unsqueeze(-1)).any(dim=-1)
            correct[k] += hit.sum().item()
        total += y.numel()

    return {k: correct[k] / total for k in ks}


@torch.no_grad()
def generation_diversity(model, tokenizer, device, num_samples=10, max_new_tokens=100, temperature=0.8, top_k=50):
    """
    distinct-n: fraction of generated n-grams that are unique. Low distinct-n
    means the model is looping / repeating itself instead of producing varied
    text. Computed across several independent generations from random starts.
    """
    model.eval()
    start_ids = torch.randint(0, len(tokenizer.vocab), (num_samples, 1), device=device)
    generated = model.generate(start_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    scores = {}
    for n in (1, 2, 3):
        all_ngrams, unique_ngrams = 0, set()
        for row in generated.tolist():
            ngrams = [tuple(row[i : i + n]) for i in range(len(row) - n + 1)]
            all_ngrams += len(ngrams)
            unique_ngrams.update(ngrams)
        scores[f"distinct-{n}"] = len(unique_ngrams) / all_ngrams if all_ngrams else 0.0

    return scores


# Common English inflections TinyStories text actually uses -- "cat" also
# needs to match "cats", "play" also needs to match "played"/"playing". Not
# a real morphological analyzer, just enough coverage for simple 3-6 letter
# story-vocabulary words to avoid undercounting a story that clearly used
# the word in a different form.
_INFLECTION_SUFFIXES = ("", "s", "es", "d", "ed", "ing")


def _word_present(word, text_lower):
    word = word.lower().strip()
    if not word:
        return True
    for suffix in _INFLECTION_SUFFIXES:
        if re.search(r"\b" + re.escape(word) + suffix + r"\b", text_lower):
            return True
    return False


@torch.no_grad()
def words_constraint_rate(model, tokenizer, prompts, device, max_new_tokens=150,
                           temperature=0.8, top_k=50, seed=0):
    """
    Experiment 3's task-specific quality metric: for each (header, words)
    prompt -- header is the "Features: ... Words: ... Story:" prefix (see
    dataset.format_instruct_prompt), words the 3 target words that prompt's
    reference story was supposed to include -- generates a continuation and
    checks whether every target word (or a simple inflection of it, see
    _word_present) shows up in it.

    Returns the fraction of prompts satisfied: 0.0 (none of the constraints
    ever get met) sets the floor an un-fine-tuned base model should sit
    near, and the fraction met by TinyStoriesInstruct's own reference
    stories sets the ceiling -- how much of that gap each fine-tuning arm
    closes is the actual result.
    """
    model.eval()
    torch.manual_seed(seed)
    hits = 0
    for header, words in prompts:
        prompt_ids = torch.tensor([tokenizer.encode(header)], device=device)
        generated = model.generate(
            prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, eos_id=tokenizer.eos_id,
        )
        # Only the newly generated continuation counts -- the header prompt
        # itself already states the target words verbatim ("Words: cat,
        # dog, ... Story:"), and generate() returns prompt+continuation
        # concatenated, so checking the whole sequence would trivially
        # "satisfy" every constraint regardless of what the model produced.
        new_tokens = generated[0, prompt_ids.shape[1]:].tolist()
        text = tokenizer.decode(new_tokens).lower()
        if all(_word_present(w, text) for w in words):
            hits += 1
    return hits / len(prompts) if prompts else 0.0


@torch.no_grad()
def throughput(model, block_size, batch_size, device, num_batches=10):
    """
    Rough forward-pass throughput (tokens/sec) -- useful for comparing the
    cost of architectural changes (layers/heads/embed_dim) on the same
    hardware, not for comparing across different machines.
    """
    model.eval()
    vocab_size = model.lm_head.out_features
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)

    sync_device(device)
    start = time.time()
    for _ in range(num_batches):
        model(x)
    sync_device(device)
    elapsed = time.time() - start

    tokens_processed = num_batches * batch_size * block_size
    return tokens_processed / elapsed


def percentile_summary(samples, percentiles=(50, 25, 75)):
    """Median + IQR (by default) of a list of timing samples -- what
    Experiments 4/5 report instead of a bare mean, since a few slow-outlier
    steps (a stray OS scheduling hiccup, a lazy CUDA kernel compile) can
    otherwise dominate a small-sample average."""
    arr = np.asarray(samples, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}


def reset_peak_memory(device):
    """Zeroes the peak-memory counter this process/device has been tracking,
    so a subsequent peak_memory_bytes() call reports the peak *since this
    call*, not since process start. No-op on backends without a resettable
    counter (MPS, CPU) -- callers there should read peak_memory_bytes()
    immediately before and after instead and take the difference/max."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device):
    """Peak (CUDA) or current (MPS driver allocation / CPU process RSS)
    memory in bytes -- the closest same-meaning number available on each
    backend. See reset_peak_memory's docstring for the CUDA-vs-other
    distinction this implies for callers."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss


@torch.no_grad()
def generation_latency(model, prompt_len, max_new_tokens, device, cache_mode,
                        num_repeats=3, top_k=50, temperature=0.8, batch_size=1):
    """
    Wall-clock latency for autoregressive generation, averaged over
    `num_repeats` runs. cache_mode selects which of model.py's three
    generation paths to use:
      "none"     -- generate(use_cache=False), recomputes full attention
                    every step
      "concat"   -- generate(use_cache=True), the original growing-by-
                    torch.cat KV cache
      "prealloc" -- generate_prealloc(...), the write-in-place KV cache
    Returns (seconds_per_run, tokens_per_sec). For the fuller Experiment 5
    benchmark (per-step timing, memory, batch size sweep, median/IQR across
    many interleaved repeats) see bench_kvcache.py -- this is the quick
    version `python3 metrics.py --kv-cache` reports.
    """
    model.eval()
    vocab_size = model.lm_head.out_features
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    def run_once():
        if cache_mode == "none":
            model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, use_cache=False)
        elif cache_mode == "concat":
            model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, use_cache=True)
        elif cache_mode == "prealloc":
            model.generate_prealloc(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        else:
            raise ValueError(f"unknown cache_mode: {cache_mode}")

    run_once()  # warmup: lazy kernel compilation / first-call overhead shouldn't count
    sync_device(device)

    start = time.time()
    for _ in range(num_repeats):
        run_once()
    sync_device(device)
    elapsed = (time.time() - start) / num_repeats

    return elapsed, max_new_tokens / elapsed


def compare_kv_cache(model, device, prompt_len=10, gen_lengths=(20, 50, 100), num_repeats=3):
    """
    Prints (and returns) a latency table comparing all three generation
    paths (no cache, concat cache, preallocated cache) at several
    generation lengths -- a quick "does KV caching matter here" check.
    Speedups are relative to "none". See bench_kvcache.py for Experiment
    5's full statistical treatment.
    """
    print(f"\n{'gen_len':>8} | {'none (s)':>10} | {'concat (s)':>11} | {'prealloc (s)':>13} | {'concat x':>9} | {'prealloc x':>11}")
    results = []
    for gen_len in gen_lengths:
        if prompt_len + gen_len > model.block_size:
            print(f"skipping gen_len={gen_len}: prompt_len + gen_len exceeds block_size={model.block_size}")
            continue

        t_none, _ = generation_latency(model, prompt_len, gen_len, device, cache_mode="none", num_repeats=num_repeats)
        t_concat, _ = generation_latency(model, prompt_len, gen_len, device, cache_mode="concat", num_repeats=num_repeats)
        t_prealloc, _ = generation_latency(model, prompt_len, gen_len, device, cache_mode="prealloc", num_repeats=num_repeats)

        print(
            f"{gen_len:>8} | {t_none:>10.4f} | {t_concat:>11.4f} | {t_prealloc:>13.4f} | "
            f"{t_none / t_concat:>8.2f}x | {t_none / t_prealloc:>10.2f}x"
        )
        results.append({
            "gen_len": gen_len,
            "no_cache_sec": t_none,
            "cache_sec": t_concat,
            "prealloc_sec": t_prealloc,
            "speedup": t_none / t_concat,
            "prealloc_speedup": t_none / t_prealloc,
        })

    return results


def run_report(checkpoint_path="checkpoints/checkpoint.pt", num_val_examples=1000, batch_size=32):
    model, tokenizer, device = load_checkpoint(checkpoint_path)
    block_size = model.block_size

    print(f"loaded checkpoint from {checkpoint_path} | device: {device}")
    print(f"vocab size: {len(tokenizer.vocab)} | params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nloading + encoding TinyStories validation split...")
    val_token_ids, val_texts = load_split_token_ids(tokenizer, "validation", num_val_examples)
    val_data = flatten(val_token_ids)

    print("\n--- perplexity / loss ---")
    avg_loss, ppl = perplexity(model, val_data, block_size, batch_size, device)
    print(f"val loss (nats/token): {avg_loss:.4f}")
    print(f"val perplexity:        {ppl:.2f}")

    bpc = bits_per_char(avg_loss, val_token_ids, val_texts)
    print(f"val bits-per-char:     {bpc:.4f}")

    baseline_ppl = unigram_baseline_perplexity(val_token_ids, len(tokenizer.vocab))
    print(f"unigram baseline ppl:  {baseline_ppl:.2f}  (model should be well below this)")

    print("\n--- next-token accuracy ---")
    acc = topk_accuracy(model, val_data, block_size, batch_size, device, ks=(1, 5))
    for k, v in acc.items():
        print(f"top-{k} accuracy: {v * 100:.2f}%")

    print("\n--- generation diversity ---")
    diversity = generation_diversity(model, tokenizer, device)
    for k, v in diversity.items():
        print(f"{k}: {v:.3f}")

    print("\n--- inference throughput ---")
    tps = throughput(model, block_size, batch_size, device)
    print(f"{tps:,.0f} tokens/sec (forward pass, batch={batch_size}, block={block_size})")


if __name__ == "__main__":
    import sys

    if "--kv-cache" in sys.argv:
        model, tokenizer, device = load_checkpoint()
        compare_kv_cache(model, device)
    else:
        run_report()
