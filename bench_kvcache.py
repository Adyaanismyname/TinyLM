"""
Experiment 5: does KV caching actually pay off at TinyLM scale, and if the
paper's original result (small, inconsistent speedups, losing outright in
5 of 9 configurations) holds up once measured properly?

Compares three generation paths from model.py -- "none" (recompute full
attention every step), "concat" (the original growing-by-torch.cat cache),
"prealloc" (write-in-place cache, model.generate_prealloc) -- at two batch
sizes and several generation lengths, with per-step timing (not just total
wall-clock) so prefill and decode cost can be told apart, and KV memory
measured both analytically and from the device. Every generation path is
already verified to produce identical output for identical input (see
tests/test_model.py); this only measures speed. Run separately on each
machine/backend (CUDA, MPS, each one's CPU) -- the paper's own candidate
explanations (bookkeeping overhead, generation length, fixed per-step
framework cost) are exactly the things that can differ by backend.

Run:
    python3 bench_kvcache.py            # full sweep
    python3 bench_kvcache.py --quick    # tiny version, to smoke-test the pipeline

Writes results/runs/e5/<device_slug>.json
"""

import argparse
import itertools
import json
import os
import random
import time

import torch

from metrics import peak_memory_bytes, percentile_summary, reset_peak_memory
from model import GPT
from train import device_description, get_device, seed_everything, sync_device

RESULTS_DIR = "results/runs/e5"

NUM_LAYERS, EMBED_DIM, NUM_HEADS = 8, 128, 8   # matches the paper's "Large" shape
VOCAB_SIZE = 1024
BLOCK_SIZE = 512
PROMPT_LEN = 16
GEN_LENGTHS = [20, 50, 100, 250, 490]
BATCH_SIZES = [1, 16]
CACHE_MODES = ["none", "concat", "prealloc"]
NUM_REPEATS = 10
HEAD_DIM = EMBED_DIM // NUM_HEADS
BYTES_PER_FLOAT = 4


def analytical_kv_bytes(batch_size, seq_len):
    """2 (K and V) x layers x seq_len x embed_dim x batch x 4 bytes (fp32) --
    the KV cache's theoretical memory footprint, independent of how it's
    implemented, to compare against what's actually measured on-device."""
    return 2 * NUM_LAYERS * seq_len * EMBED_DIM * batch_size * BYTES_PER_FLOAT


@torch.no_grad()
def timed_generate(model, cache_mode, batch_size, prompt_len, gen_len, device):
    """Runs one generation, greedy (temperature=0, so every cache mode
    produces literally the same tokens -- speed, not diversity, is what's
    being measured), returning (total_seconds, peak_memory_bytes)."""
    prompt = torch.randint(0, VOCAB_SIZE, (batch_size, prompt_len), device=device)

    reset_peak_memory(device)
    sync_device(device)
    start = time.time()

    if cache_mode == "none":
        model.generate(prompt, max_new_tokens=gen_len, temperature=0)
    elif cache_mode == "concat":
        model.generate(prompt, max_new_tokens=gen_len, temperature=0, use_cache=True)
    elif cache_mode == "prealloc":
        model.generate_prealloc(prompt, max_new_tokens=gen_len, temperature=0)
    else:
        raise ValueError(f"unknown cache_mode: {cache_mode}")

    sync_device(device)
    elapsed = time.time() - start
    return elapsed, peak_memory_bytes(device)


def run(quick=False, out_dir=RESULTS_DIR):
    device = get_device()
    print(f"bench_kvcache: device={device_description(device)}", flush=True)

    seed_everything(0)
    model = GPT(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, embed_dim=EMBED_DIM,
                num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=0.0).to(device)
    model.eval()

    gen_lengths = [20, 50] if quick else GEN_LENGTHS
    batch_sizes = [1] if quick else BATCH_SIZES
    num_repeats = 2 if quick else NUM_REPEATS

    configs = [
        {"cache_mode": m, "batch_size": b, "gen_len": g}
        for m, b, g in itertools.product(CACHE_MODES, batch_sizes, gen_lengths)
        if PROMPT_LEN + g <= BLOCK_SIZE
    ]

    # One warmup generation per (cache_mode, batch_size, gen_len) combo
    # before any timed repeat, so lazy kernel compilation / first-call
    # overhead isn't charged to whichever config happens to run first.
    for cfg in configs:
        timed_generate(model, cfg["cache_mode"], cfg["batch_size"], PROMPT_LEN, cfg["gen_len"], device)

    trials = configs * num_repeats
    random.shuffle(trials)  # interleaved, so thermal/scheduling drift can't correlate with cache_mode

    raw = {}  # (cache_mode, batch_size, gen_len) -> list of (seconds, peak_mem_bytes)
    for i, cfg in enumerate(trials):
        key = (cfg["cache_mode"], cfg["batch_size"], cfg["gen_len"])
        if i % 10 == 0:
            print(f"[{i + 1}/{len(trials)}] {key}", flush=True)
        seconds, peak_mem = timed_generate(model, cfg["cache_mode"], cfg["batch_size"], PROMPT_LEN, cfg["gen_len"], device)
        raw.setdefault(key, []).append((seconds, peak_mem))

    results = []
    for (cache_mode, batch_size, gen_len), samples in raw.items():
        seconds = [s for s, _ in samples]
        peak_mems = [m for _, m in samples]
        results.append({
            "cache_mode": cache_mode, "batch_size": batch_size, "gen_len": gen_len,
            "prompt_len": PROMPT_LEN, "seconds": percentile_summary(seconds),
            "tokens_per_sec_p50": gen_len / percentile_summary(seconds)["p50"],
            "peak_memory_bytes_p50": sorted(peak_mems)[len(peak_mems) // 2],
            "analytical_kv_bytes": analytical_kv_bytes(batch_size, PROMPT_LEN + gen_len) if cache_mode != "none" else 0,
            "num_samples": len(samples),
        })

    device_slug = device_description(device).replace(" ", "_").replace("/", "_")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{device_slug}.json")
    with open(path, "w") as f:
        json.dump({
            "device": device_description(device), "torch_version": torch.__version__,
            "model_config": {"num_layers": NUM_LAYERS, "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS,
                              "block_size": BLOCK_SIZE, "vocab_size": VOCAB_SIZE},
            "prompt_len": PROMPT_LEN, "num_repeats": num_repeats, "results": results,
        }, f, indent=2)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Tiny sweep for smoke-testing the pipeline.")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    run(quick=args.quick, out_dir=args.out_dir)
