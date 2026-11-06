"""
Task 5 (targeted check, no training of real models): is the width-1024 LoRA
"crossover" in Experiment 4 explained by full fine-tuning running out of GPU
memory? Full fine-tuning at width 1024, batch 16 peaked at 5,815 MiB allocated
on a 6,144 MiB card. Re-time full FT vs LoRA r8 at width 1024 with batch 16
(as in E4) and batch 8 (about half the activation memory), interleaved, using
bench_train.run_one unchanged. If the ~1.8x LoRA advantage persists at batch
8, memory pressure is not the explanation; if it collapses, it was.

  python -B camera_ready/bench_mem_check.py      (run with the GPU otherwise idle)
"""

import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, save_json  # noqa: E402

sys.path.insert(0, ROOT)
from bench_train import METHODS, run_one  # noqa: E402
from train import get_device, seed_everything  # noqa: E402

WARMUP, TIMED, REPEATS = 15, 40, 2


def main():
    device = get_device()
    free, total = torch.cuda.mem_get_info()
    methods = [m for m in METHODS if m["name"] in ("full", "lora_r8")]
    combos = [(b, m) for b in (16, 8) for m in methods] * REPEATS
    seed_everything(0)
    random.shuffle(combos)
    rows = []
    for i, (b, m) in enumerate(combos):
        r = run_one(1024, m, device, b, WARMUP, TIMED)
        rows.append({"batch_size": b, "method": m["name"], "median_s": r["summary"]["p50"],
                     "iqr_s": [r["summary"]["p25"], r["summary"]["p75"]],
                     "peak_alloc_MiB": round(r["peak_memory_bytes"] / 2**20),
                     "s_per_sample": r["summary"]["p50"] / b})
        print(i, rows[-1], flush=True)
        torch.cuda.empty_cache()
    out = {"device_total_MiB": round(total / 2**20), "free_at_start_MiB": round(free / 2**20),
           "warmup": WARMUP, "timed": TIMED, "repeats": REPEATS, "runs": rows, "speedup": {}}
    for b in (16, 8):
        f = sorted(r["median_s"] for r in rows if r["batch_size"] == b and r["method"] == "full")
        l = sorted(r["median_s"] for r in rows if r["batch_size"] == b and r["method"] == "lora_r8")
        out["speedup"][b] = {"full_medians": f, "lora_r8_medians": l,
                             "ratio_of_means": (sum(f) / len(f)) / (sum(l) / len(l))}
    print(out["speedup"])
    save_json(out, "bench_mem_check.json")


if __name__ == "__main__":
    main()
