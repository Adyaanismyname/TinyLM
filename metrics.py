"""
Evaluation metrics for the GPT model trained by train.py.

Loads checkpoints/checkpoint.pt directly (model weights + the exact
vocab/merges/unk token it was trained with) rather than rebuilding the
dataset pipeline, so the tokenizer here always matches the checkpointed
model instead of accidentally retraining a new BPE vocab.

Computes:
  - perplexity / cross-entropy loss, evaluated over every window of the
    validation set (train.py's estimate_loss only samples random windows,
    which is fine for a training-time progress readout but not precise
    enough for a final benchmark number)
  - bits-per-character, which normalizes loss by character count instead
    of token count -- useful because it stays comparable even if you change
    vocab_size or retrain the tokenizer, unlike perplexity
  - a unigram frequency baseline, so you can tell whether the model has
    actually learned something beyond raw token frequency
  - top-1 / top-5 next-token accuracy
  - generation diversity (distinct-n), to catch degenerate/repetitive output
  - inference throughput (tokens/sec)

Run standalone:

    python3 metrics.py
"""

import math
import time
from collections import Counter

import torch
import torch.nn.functional as F

from BPE import BPETokenizer
from dataset import preprocess_text
from model import GPT
from train import get_device


def load_checkpoint(path="checkpoints/checkpoint.pt"):
    device = get_device()
    checkpoint = torch.load(path, map_location=device)

    tokenizer = BPETokenizer(
        checkpoint["vocab"],
        checkpoint["merges"],
        unk_token=checkpoint.get("unk_token", "<unk>"),
    )

    model = GPT(vocab_size=len(tokenizer.vocab), **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device


def load_split_token_ids(tokenizer, split, num_examples):
    """
    Loads a TinyStories split and encodes it with an already-trained
    tokenizer (no BPE retraining), so results line up with the checkpointed
    model. Returns (token_id_lists, raw_texts) -- the raw texts are needed
    for bits-per-character.
    """
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories")
    texts = [preprocess_text(t) for t in dataset[split]["text"][:num_examples]]
    token_ids = [tokenizer.encode(t) for t in texts]
    return token_ids, texts


def flatten(token_id_lists):
    flat = [tid for story in token_id_lists for tid in story]
    return torch.tensor(flat, dtype=torch.long)


def _windows(data, block_size, batch_size):
    """Yields (x, y) batches covering every non-overlapping window in data exactly once."""
    num_windows = (len(data) - 1) // block_size
    for start in range(0, num_windows, batch_size):
        idxs = range(start, min(start + batch_size, num_windows))
        x = torch.stack([data[i * block_size : i * block_size + block_size] for i in idxs])
        y = torch.stack([data[i * block_size + 1 : i * block_size + 1 + block_size] for i in idxs])
        yield x, y


@torch.no_grad()
def perplexity(model, data, block_size, batch_size, device):
    """Exact (non-sampled) average cross-entropy loss and perplexity over all of `data`."""
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for x, y in _windows(data, block_size, batch_size):
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        n = x.numel()
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

    for x, y in _windows(data, block_size, batch_size):
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

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    start = time.time()
    for _ in range(num_batches):
        model(x)
    sync()
    elapsed = time.time() - start

    tokens_processed = num_batches * batch_size * block_size
    return tokens_processed / elapsed


@torch.no_grad()
def generation_latency(model, prompt_len, max_new_tokens, device, use_cache, num_repeats=3, top_k=50, temperature=0.8):
    """
    Wall-clock latency for a single autoregressive generation (batch=1,
    matching real single-request serving), with vs without the KV cache
    added in model.py. Returns (seconds_per_run, tokens_per_sec).
    """
    model.eval()
    vocab_size = model.lm_head.out_features
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # One warmup run so lazy kernel compilation / first-call overhead isn't
    # counted against either method.
    model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, use_cache=use_cache)
    sync()

    start = time.time()
    for _ in range(num_repeats):
        model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, use_cache=use_cache)
    sync()
    elapsed = (time.time() - start) / num_repeats

    return elapsed, max_new_tokens / elapsed


def compare_kv_cache(model, device, prompt_len=10, gen_lengths=(20, 50, 100), num_repeats=3):
    """
    Prints (and returns) a latency table comparing generate(use_cache=False)
    vs generate(use_cache=True) at several generation lengths -- the numbers
    behind a "does KV caching actually matter here" article section.
    """
    print(f"\n{'gen_len':>8} | {'no-cache (s)':>13} | {'cached (s)':>11} | {'speedup':>8}")
    results = []
    for gen_len in gen_lengths:
        if prompt_len + gen_len > model.block_size:
            print(f"skipping gen_len={gen_len}: prompt_len + gen_len exceeds block_size={model.block_size}")
            continue

        t_nocache, _ = generation_latency(model, prompt_len, gen_len, device, use_cache=False, num_repeats=num_repeats)
        t_cache, _ = generation_latency(model, prompt_len, gen_len, device, use_cache=True, num_repeats=num_repeats)
        speedup = t_nocache / t_cache

        print(f"{gen_len:>8} | {t_nocache:>13.4f} | {t_cache:>11.4f} | {speedup:>7.2f}x")
        results.append({
            "gen_len": gen_len,
            "no_cache_sec": t_nocache,
            "cache_sec": t_cache,
            "speedup": speedup,
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
