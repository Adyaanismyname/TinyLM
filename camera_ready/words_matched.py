"""
Task 2: a like-for-like Words-constraint comparison.

  python -B camera_ready/words_matched.py               # 150-token budget (the E3 protocol)
  python -B camera_ready/words_matched.py --budget 0    # 0 = fill the 512-token context (supplementary)

1. Reconstructs exactly which Instruct test example each of the 500 prompts
   in data/words_eval_prompts.json came from (prepare_data.build_words_eval_prompts
   is deterministic), so every prompt has its own reference story.
2. Scores the reference story truncated to the same token budget the models
   got, with the unchanged matcher (metrics._word_present).
3. Regenerates the seed-0 outputs of every saved checkpoint with the exact
   E3 procedure (metrics.words_constraint_rate: seed 0, T=0.8, top-k 50,
   EOS stop) and keeps the text, so prompt-level results exist; the stored
   5-seed rates come from results/runs/e3/*.json.
No training happens here.
"""

import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT, ROOT, RUNS, ids, mask, save_json, story_table  # noqa: E402

sys.path.insert(0, ROOT)
from metrics import _word_present, load_checkpoint  # noqa: E402

CKPTS = {"baseline_frozen": "checkpoints/e2_base_layers8_seed0.pt",
         "lora_r1": "checkpoints/e3_lora_r1_seed0.pt",
         "lora_r8": "checkpoints/e3_lora_r8_seed0.pt",
         "lora_r64": "checkpoints/e3_lora_r64_seed0.pt",
         "full_finetune": "checkpoints/e3_full_finetune_seed0.pt"}
ARMS = ["baseline_frozen", "lora_r1", "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32", "lora_r64", "full_finetune"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def reconstruct_prompt_examples(tokenizer, prompts):
    a = np.asarray(ids("instruct_test"))
    tab = story_table("instruct_test")
    headers = [tokenizer.decode(a[f:h].tolist()) for f, h in zip(tab["first"], tab["hdr_end"])]
    with_words = [i for i, hd in enumerate(headers) if re.search(r"(^|\s)Words: \S", hd)]
    order = list(range(len(with_words)))
    random.Random(0).shuffle(order)                         # dataset.split_dev(seed=0)
    picked = [with_words[i] for i in sorted(set(order[:len(prompts)]))]
    for (hdr, _), ex in zip(prompts, picked):               # every prompt must match its example exactly
        assert tokenizer.encode(hdr) == a[tab["first"][ex]:tab["hdr_end"][ex]].tolist(), ex
    return a, tab, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150, help="max new tokens; 0 = fill the context window")
    args = ap.parse_args()

    prompts = json.load(open(os.path.join(ROOT, "data", "words_eval_prompts.json")))
    model0, tok, device = load_checkpoint(CKPTS["baseline_frozen"])
    a, tab, picked = reconstruct_prompt_examples(tok, prompts)
    print(f"all {len(prompts)} prompts matched to their own reference example", flush=True)

    budgets = {}
    ref = []
    for (hdr, words), ex in zip(prompts, picked):
        story = a[tab["hdr_end"][ex]:tab["eos"][ex]].tolist()   # leading-space token included, EOS excluded
        plen = len(tok.encode(hdr))
        budget = args.budget if args.budget > 0 else 512 - plen
        budgets[ex] = budget
        r = {"example": int(ex), "words": words, "ref_len": len(story), "budget": budget}
        for label, n in (("budget", budget), ("150", 150), ("300", 300), ("full", len(story))):
            text = tok.decode(story[:n]).lower()
            present = [_word_present(w, text) for w in words]
            r[f"ref_{label}_present"] = present
        ref.append(r)

    def ref_rate(label):
        k = sum(all(r[f"ref_{label}_present"]) for r in ref)
        return {"hits": k, "n": len(ref), "pct": 100 * k / len(ref), "wilson95": wilson(k, len(ref)),
                "mean_words_present": float(np.mean([sum(r[f"ref_{label}_present"]) for r in ref]))}

    result = {"budget_arg": args.budget,
              "reference": {lab: ref_rate(lab) for lab in ("budget", "150", "300", "full")},
              "reference_len_tokens": {"mean": float(np.mean([r["ref_len"] for r in ref])),
                                       "median": float(np.median([r["ref_len"] for r in ref]))}}

    # stored 5-seed rates (always on the same 500 prompts, 150-token budget)
    stored = {}
    for arm in ARMS:
        vals = []
        for s in range(5):
            p = os.path.join(RUNS, "e3", f"{arm}_seed{s}.json")
            vals.append(round(json.load(open(p))["final_metrics"]["words_constraint_rate"] * 500))
        stored[arm] = {"hits_per_seed": vals, "mean_pct": 100 * float(np.mean(vals)) / 500,
                       "pooled_wilson95": wilson(sum(vals), 2500)}
    result["stored_150_token_rates"] = stored

    # regenerate seed-0 outputs with the exact E3 procedure
    gen = {}
    for arm, path in CKPTS.items():
        model, _, _ = load_checkpoint(path)
        model.eval()
        torch.manual_seed(0)
        outs = []
        for (hdr, words), ex in zip(prompts, picked):
            pid = torch.tensor([tok.encode(hdr)], device=device)
            g = model.generate(pid, max_new_tokens=budgets[ex], temperature=0.8, top_k=50,
                               eos_id=tok.eos_id, use_cache=args.budget == 0)
            new = g[0, pid.shape[1]:].tolist()
            text = tok.decode(new).lower()
            outs.append({"present": [_word_present(w, text) for w in words],
                         "n_new_tokens": len(new), "hit_eos": tok.eos_id in new,
                         "text": tok.decode([t for t in new if t != tok.eos_id])})
        k = sum(all(o["present"]) for o in outs)
        gen[arm] = outs
        rr = result["reference"]["budget"]
        b = sum(1 for o, r in zip(outs, ref) if all(r["ref_budget_present"]) and not all(o["present"]))
        c = sum(1 for o, r in zip(outs, ref) if all(o["present"]) and not all(r["ref_budget_present"]))
        result.setdefault("regenerated_seed0", {})[arm] = {
            "hits": k, "pct": 100 * k / len(outs), "wilson95": wilson(k, len(outs)),
            "stored_seed0_hits": stored[arm]["hits_per_seed"][0] if args.budget == 150 else None,
            "mean_words_present": float(np.mean([sum(o["present"]) for o in outs])),
            "at_least_1_word_pct": 100 * float(np.mean([any(o["present"]) for o in outs])),
            "at_least_2_words_pct": 100 * float(np.mean([sum(o["present"]) >= 2 for o in outs])),
            "per_word_presence_pct": 100 * float(np.mean([p for o in outs for p in o["present"]])),
            "stopped_at_eos_pct": 100 * float(np.mean([o["hit_eos"] for o in outs])),
            "vs_reference_mcnemar": {"ref_only": b, "model_only": c, "p": mcnemar_exact(b, c)},
        }
        print(arm, result["regenerated_seed0"][arm], flush=True)
        del model
        torch.cuda.empty_cache()
    rr = result["reference"]["budget"]
    result["reference_budget_partial"] = {
        "at_least_1_word_pct": 100 * float(np.mean([any(r["ref_budget_present"]) for r in ref])),
        "at_least_2_words_pct": 100 * float(np.mean([sum(r["ref_budget_present"]) >= 2 for r in ref])),
        "per_word_presence_pct": 100 * float(np.mean([p for r in ref for p in r["ref_budget_present"]]))}

    tag = "150" if args.budget == 150 else ("fill" if args.budget == 0 else str(args.budget))
    save_json(result, f"words_matched_{tag}.json")
    with open(os.path.join(OUT, f"words_outputs_seed0_{tag}.json"), "w") as f:
        json.dump({"prompts": [{"example": r["example"], "words": r["words"], "header": p[0]} for r, p in zip(ref, prompts)],
                   "reference_present": [r["ref_budget_present"] for r in ref],
                   "outputs": gen}, f, indent=1)
    print(f"wrote camera_ready/out/words_outputs_seed0_{tag}.json")


if __name__ == "__main__":
    main()
