"""
Experiment 4: does LoRA actually train faster than full fine-tuning, and if
so, at what model width does the crossover happen?

The paper's original result (every LoRA rank trained *slower* than full
fine-tuning) was measured inside a quality run whose timing also included
periodic evaluation passes -- see train.py's module docstring for why that
confounds "which method is faster" with "how many eval checkpoints did this
particular run happen to hit". This is a standalone benchmark instead:
forward + backward + optimizer-step wall-clock time and peak memory, on
random weights and random data (speed doesn't depend on what the weights
have learned), across a width sweep, run separately on each machine/backend
this project uses -- CUDA, MPS, and each one's CPU -- since the paper's own
explanation for its result was per-step hardware overhead, and the only way
to test that is to change the hardware.

Run:
    python3 bench_train.py            # full sweep (widths x methods x repeats)
    python3 bench_train.py --quick    # tiny version, to smoke-test the pipeline

Writes results/runs/e4/<device_slug>.json
"""

import argparse
import itertools
import json
import os
import random
import time

import torch

from lora import add_lora, trainable_parameters
from metrics import peak_memory_bytes, percentile_summary, reset_peak_memory
from model import GPT
from train import build_optimizer, device_description, get_device, seed_everything, sync_device

RESULTS_DIR = "results/runs/e4"

WIDTHS = [64, 128, 256, 512, 1024]   # embed_dim; head_dim fixed at 64 (num_heads = width // 64)
NUM_LAYERS = 8
METHODS = [
    {"name": "full", "mode": "full", "r": None},
    {"name": "lora_r8", "mode": "lora", "r": 8},
    {"name": "lora_r64", "mode": "lora", "r": 64},
]
VOCAB_SIZE = 1024
BLOCK_SIZE = 512
BATCH_SIZE = 16
WARMUP_STEPS = 50
TIMED_STEPS = 200
NUM_REPEATS = 5
GRAD_CLIP = 1.0
LEARNING_RATE = 3e-4  # irrelevant to speed, just needs a valid optimizer


def build_model(width, mode, r, device):
    num_heads = max(1, width // 64)
    model = GPT(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, embed_dim=width,
                num_heads=num_heads, num_layers=NUM_LAYERS, dropout=0.0).to(device)
    if mode == "lora":
        add_lora(model, r=r, alpha=r * 2)
    return model


def run_one(width, method, device, batch_size, warmup_steps, timed_steps):
    model = build_model(width, method["mode"], method.get("r"), device)
    model.train()
    optimizer = build_optimizer(model, learning_rate=LEARNING_RATE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    x = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)

    def step():
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()

    for _ in range(warmup_steps):
        step()
    sync_device(device)

    reset_peak_memory(device)
    step_times = []
    for _ in range(timed_steps):
        sync_device(device)
        t0 = time.time()
        step()
        sync_device(device)
        step_times.append(time.time() - t0)
    peak_mem = peak_memory_bytes(device)

    trainable, total = trainable_parameters(model)
    return {
        "width": width, "method": method["name"], "trainable_params": trainable, "total_params": total,
        "step_times_sec": step_times, "summary": percentile_summary(step_times),
        "peak_memory_bytes": peak_mem,
    }


def run(quick=False, batch_size=BATCH_SIZE, out_dir=RESULTS_DIR):
    device = get_device()
    print(f"bench_train: device={device_description(device)}", flush=True)

    widths = [128, 256] if quick else WIDTHS
    warmup_steps = 5 if quick else WARMUP_STEPS
    timed_steps = 10 if quick else TIMED_STEPS
    num_repeats = 1 if quick else NUM_REPEATS

    combos = list(itertools.product(widths, METHODS)) * num_repeats
    seed_everything(0)
    random.shuffle(combos)  # interleaved order so drift over the run (thermal throttling,
                             # OS scheduling) doesn't systematically favor whichever method
                             # happens to run first/last

    results = []
    for i, (width, method) in enumerate(combos):
        print(f"[{i + 1}/{len(combos)}] width={width} method={method['name']}", flush=True)
        try:
            results.append(run_one(width, method, device, batch_size, warmup_steps, timed_steps))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"  OOM at width={width} method={method['name']} batch_size={batch_size} -- skipping")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    device_slug = device_description(device).replace(" ", "_").replace("/", "_")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{device_slug}.json")
    with open(path, "w") as f:
        json.dump({
            "device": device_description(device), "torch_version": torch.__version__,
            "batch_size": batch_size, "block_size": BLOCK_SIZE, "num_layers": NUM_LAYERS,
            "warmup_steps": warmup_steps, "timed_steps": timed_steps, "num_repeats": num_repeats,
            "results": results,
        }, f, indent=2)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Tiny sweep for smoke-testing the pipeline.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    run(quick=args.quick, batch_size=args.batch_size, out_dir=args.out_dir)
