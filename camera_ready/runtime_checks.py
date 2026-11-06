"""
Task 5: runtime findings, re-read from the stored benchmark JSONs, plus a
cache-equivalence check at the exact Experiment 5 benchmark configuration.

  python -B camera_ready/runtime_checks.py

(The unit tests in tests/test_model.py cover small models; this repeats the
check for the 8-layer / 128-dim model, batch 16, 16-token prompt, 490 greedy
tokens, random weights -- the largest E5 configuration.)
"""

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, RUNS, save_json  # noqa: E402

sys.path.insert(0, ROOT)
from model import GPT  # noqa: E402


@torch.no_grad()
def cache_equivalence():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = GPT(vocab_size=1024, block_size=512, embed_dim=128, num_heads=8, num_layers=8, dropout=0.0).to(dev).eval()
    prompt = torch.randint(0, 1024, (16, 16), device=dev)
    a = m.generate(prompt, max_new_tokens=490, temperature=0)
    b = m.generate(prompt, max_new_tokens=490, temperature=0, use_cache=True)
    c = m.generate_prealloc(prompt, max_new_tokens=490, temperature=0)
    # logits of the full 506-token sequence: one uncached forward vs. step-by-step cached decoding
    full, _, _ = m(a)
    logits_cat = []
    lg, _, cache = m(a[:, :16], use_cache=True)
    logits_cat.append(lg)
    for t in range(16, a.shape[1]):
        lg, _, cache = m(a[:, t:t + 1], kv_cache=cache, use_cache=True)
        logits_cat.append(lg)
    cat = torch.cat(logits_cat, dim=1)
    pre = m.init_prealloc_cache(16, dev)
    logits_pre = [m.forward_prealloc(a[:, :16], pre, 0)]
    for t in range(16, a.shape[1]):
        logits_pre.append(m.forward_prealloc(a[:, t:t + 1], pre, t))
    pre_l = torch.cat(logits_pre, dim=1)
    return {"device": str(dev), "tokens_equal_none_vs_concat": bool(torch.equal(a, b)),
            "tokens_equal_none_vs_prealloc": bool(torch.equal(a, c)),
            "max_abs_logit_diff_concat": float((full - cat).abs().max()),
            "max_abs_logit_diff_prealloc": float((full - pre_l).abs().max()),
            "logit_scale_max_abs": float(full.abs().max())}


def e4_tables():
    e4 = json.load(open(glob.glob(os.path.join(RUNS, "e4", "*.json"))[0]))
    tab = {}
    for r in e4["results"]:
        tab.setdefault((r["width"], r["method"]), []).append(r)
    out = {}
    for w in sorted({k[0] for k in tab}):
        row = {}
        for meth in ("full", "lora_r8", "lora_r64"):
            rs = tab[(w, meth)]
            row[meth] = {"per_repeat_median_s": [round(x["summary"]["p50"], 4) for x in rs],
                         "pooled_median_s": float(np.median(np.concatenate([x["step_times_sec"] for x in rs]))),
                         "peak_alloc_GB": round(rs[0]["peak_memory_bytes"] / 1e9, 2),
                         "peak_alloc_MiB": round(rs[0]["peak_memory_bytes"] / 2**20)}
        f = row["full"]["per_repeat_median_s"]
        for meth in ("lora_r8", "lora_r64"):
            l = row[meth]["per_repeat_median_s"]
            row[meth]["speedup_pooled"] = row["full"]["pooled_median_s"] / row[meth]["pooled_median_s"]
            # every full-vs-LoRA pairing of repeat medians: how often is LoRA faster?
            ratios = np.array([fi / li for fi in f for li in l])
            row[meth]["speedup_all_pairs_min_max"] = [float(ratios.min()), float(ratios.max())]
            row[meth]["lora_faster_in_pairs_pct"] = float(100 * (ratios > 1).mean())
        out[w] = row
    return {"gpu_total_MiB": 6144, "rows": out}


def e5_tables():
    e5 = json.load(open(glob.glob(os.path.join(RUNS, "e5", "*.json"))[0]))
    t = {(r["cache_mode"], r["batch_size"], r["gen_len"]): r for r in e5["results"]}
    out = {}
    for b in (1, 16):
        for g in (20, 50, 100, 250, 490):
            n = t[("none", b, g)]
            row = {"none_s": n["seconds"]}
            for mode in ("concat", "prealloc"):
                c = t[(mode, b, g)]
                row[mode] = {"s": c["seconds"], "speedup_median": n["seconds"]["p50"] / c["seconds"]["p50"],
                             "peak_MB": round(c["peak_memory_bytes_p50"] / 1e6, 1)}
            row["none_peak_MB"] = round(n["peak_memory_bytes_p50"] / 1e6, 1)
            row["kv_analytic_MB"] = round(t[("concat", b, g)]["analytical_kv_bytes"] / 1e6, 1)
            out[f"b{b}_g{g}"] = row
    return out


def main():
    res = {"cache_equivalence_e5_config": cache_equivalence(), "e4": e4_tables(), "e5": e5_tables()}
    print(json.dumps(res["cache_equivalence_e5_config"], indent=1))
    save_json(res, "runtime_checks.json")


if __name__ == "__main__":
    main()
