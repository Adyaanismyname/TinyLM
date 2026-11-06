"""
Task 1b: re-evaluate every saved model on the clean test subset, and
measure what the overlap actually did (no training).

  python -B camera_ready/clean_eval.py     (after camera_ready/contamination.py)

Per-token losses are computed with exactly the original evaluation windows
(first 2M TinyStories test tokens for E1/E2; first 1M tokens of each test
set for E3, Instruct loss-masked to story tokens), so the "all" numbers must
reproduce the stored JSON values; subsets are then just partitions of those
same token losses by the test story each token belongs to.

Models available without retraining: the E2 8-layer seed-0 model (= the E3
base / frozen baseline) and the seed-0 E3 models for LoRA r1, r8, r64 and
full fine-tuning (checkpoints/). Everything else was never saved.
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import BS, OUT, ROOT, RUNS, ids, mask, save_json  # noqa: E402

sys.path.insert(0, ROOT)
from metrics import load_checkpoint  # noqa: E402

CKPTS = {"base": "checkpoints/e2_base_layers8_seed0.pt",
         "lora_r1": "checkpoints/e3_lora_r1_seed0.pt",
         "lora_r8": "checkpoints/e3_lora_r8_seed0.pt",
         "lora_r64": "checkpoints/e3_lora_r64_seed0.pt",
         "full_finetune": "checkpoints/e3_full_finetune_seed0.pt"}
SETS = {"e12_tinystories_test": ("tinystories_test", 2_000_000, False),
        "e3_tinystories_test": ("tinystories_test", 1_000_000, False),
        "e3_instruct_test": ("instruct_test", 1_000_000, True)}
RNG = np.random.default_rng(0)


@torch.no_grad()
def token_losses(model, data, m, device, batch_size=32):
    nwin = (len(data) - 1) // BS
    out = np.empty(nwin * BS, dtype=np.float64)   # out[p-1] = loss of target position p
    for s in range(0, nwin, batch_size):
        w = np.arange(s, min(s + batch_size, nwin))
        x = torch.from_numpy(np.stack([np.asarray(data[i * BS:i * BS + BS]) for i in w]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([np.asarray(data[i * BS + 1:i * BS + BS + 1]) for i in w]).astype(np.int64)).to(device)
        logits, _, _ = model(x)
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        out[s * BS:(s + len(w)) * BS] = l.double().cpu().numpy()
    counted = np.ones(nwin * BS, dtype=bool) if m is None else np.asarray(m[1:1 + nwin * BS]).astype(bool)
    return out, counted


def group_stats(loss, counted, owner, story_mask, boot=0):
    sel = counted & story_mask[owner]
    n = int(sel.sum())
    if n == 0:
        return {"tokens": 0}
    mean = float(loss[sel].mean())
    r = {"tokens": n, "stories": int(np.unique(owner[sel]).size), "loss": mean, "ppl": math.exp(mean)}
    if boot:
        # story-level bootstrap of the token-weighted mean
        st = owner[sel]
        uniq, inv = np.unique(st, return_inverse=True)
        sums = np.bincount(inv, weights=loss[sel])
        cnts = np.bincount(inv)
        bs = []
        for _ in range(boot):
            k = RNG.integers(0, len(uniq), len(uniq))
            bs.append(sums[k].sum() / cnts[k].sum())
        r["loss_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        r["_boot"] = bs
    return r


def diff_ci(a, b):
    d = np.array(a["_boot"]) - np.array(b["_boot"])
    return {"loss_diff": a["loss"] - b["loss"], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def strip(d):
    return {k: (strip(v) if isinstance(v, dict) else v) for k, v in d.items() if k != "_boot"}


def main():
    con = {t: dict(np.load(os.path.join(OUT, f"contamination_{t}.npz"))) for t in ("tinystories_test", "instruct_test")}
    stored = {
        ("e12_tinystories_test", "base"): json.load(open(os.path.join(RUNS, "e2", "layers8_seed0.json")))["final_metrics"]["val_loss"],
    }
    for arm, run in (("base", "baseline_frozen"), ("lora_r1", "lora_r1"), ("lora_r8", "lora_r8"),
                     ("lora_r64", "lora_r64"), ("full_finetune", "full_finetune")):
        fm = json.load(open(os.path.join(RUNS, "e3", f"{run}_seed0.json")))["final_metrics"]
        stored[("e3_instruct_test", arm)] = fm["instruct_val_loss"]
        stored[("e3_tinystories_test", arm)] = fm["tinystories_forget_val_loss"]

    results = {"note": "loss = mean nats/token over scored tokens; ppl = exp(loss). Seed-0 models only.", "sets": {}}
    for key, (tname, limit, masked) in SETS.items():
        c = con[tname]
        data = ids(tname)[:limit]
        m = mask(tname)[:limit] if masked else None
        nwin = (limit - 1) // BS
        pos = np.arange(1, nwin * BS + 1)
        owner = np.searchsorted(c["first"], pos, side="right") - 1
        exact = c["exact_tinystories_train"] | c["exact_instruct_train"]
        groups = {
            "all": np.ones(len(c["first"]), dtype=bool),
            "clean_exact": ~exact,
            "clean_exact_no50gram": ~exact & (c["ngram50_frac"] == 0),
            "clean_exact_no100gram": ~exact & (c["ngram100_frac"] == 0),
            "exact_dup_tinystories_train": c["exact_tinystories_train"],
        }
        ex = c["exposure_e2_seed0"]
        dts = c["exact_tinystories_train"]
        any_exposure = ex > 0
        if tname == "tinystories_test" and key.startswith("e3"):
            any_exposure = any_exposure | (c["exposure_e3ft_seed0"] > 0)   # copies trained on during seed-0 fine-tuning
        groups["not_exposed"] = ~any_exposure     # drop only stories whose copy the evaluated models trained on
        groups["dup_unexposed_by_base"] = dts & (ex == 0)
        groups["dup_partly_exposed_by_base"] = dts & (ex > 0) & (ex < 1)
        groups["dup_exposed_1to2_passes_by_base"] = dts & (ex >= 1) & (ex < 2)
        groups["dup_exposed_ge1_by_base"] = dts & (ex >= 1)
        if tname == "tinystories_test":
            groups["in_instruct_train"] = c["exact_instruct_train"]
            groups["in_instruct_train_exposed_by_ft_seed0"] = c["exact_instruct_train"] & (c["exposure_e3ft_seed0"] > 0)

        arms = ["base"] if key == "e12_tinystories_test" else list(CKPTS)
        res = {}
        for arm in arms:
            model, _, device = load_checkpoint(CKPTS[arm])
            loss, counted = token_losses(model, data, m, device)
            g = {name: group_stats(loss, counted, owner, sm, boot=1000 if name.startswith(("dup_", "clean_exact", "all", "not_exposed")) else 0)
                 for name, sm in groups.items()}
            g["reproduces_stored"] = {"stored_loss": stored[(key, arm)], "recomputed_loss": g["all"]["loss"],
                                      "abs_diff": abs(stored[(key, arm)] - g["all"]["loss"])}
            g["memorization_gap_exposed_minus_unexposed"] = diff_ci(g["dup_exposed_ge1_by_base"], g["dup_unexposed_by_base"])
            g["memorization_gap_1to2_passes_minus_unexposed"] = diff_ci(g["dup_exposed_1to2_passes_by_base"], g["dup_unexposed_by_base"])
            g["clean_minus_all"] = diff_ci(g["clean_exact_no50gram"], g["all"])
            res[arm] = g
            print(key, arm, {k: round(v.get("ppl", float("nan")), 4) for k, v in g.items() if isinstance(v, dict) and "ppl" in v},
                  "repro diff", g["reproduces_stored"]["abs_diff"], flush=True)
            del model
            torch.cuda.empty_cache()
        results["sets"][key] = {arm: strip(v) for arm, v in res.items()}

        if key == "e3_instruct_test":
            rec = {}
            for sub in ("all", "not_exposed", "clean_exact", "clean_exact_no50gram", "clean_exact_no100gram"):
                b, f = res["base"][sub]["ppl"], res["full_finetune"][sub]["ppl"]
                rec[sub] = {arm: (b - res[arm][sub]["ppl"]) / (b - f) for arm in ("lora_r1", "lora_r8", "lora_r64")}
                rec[sub]["base_ppl"], rec[sub]["full_ppl"] = b, f
            results["e3_recovery_seed0"] = rec
        if key == "e3_tinystories_test":
            fg = {}
            for sub in ("all", "not_exposed", "clean_exact", "clean_exact_no50gram", "clean_exact_no100gram"):
                b = res["base"][sub]["ppl"]
                fg[sub] = {arm: res[arm][sub]["ppl"] - b for arm in ("lora_r1", "lora_r8", "lora_r64", "full_finetune")}
                fg[sub]["base_ppl"] = b
            results["e3_forgetting_seed0_ppl_increase"] = fg
    save_json(results, "clean_eval.json")


if __name__ == "__main__":
    main()
