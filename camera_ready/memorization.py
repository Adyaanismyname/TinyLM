"""
Task 1d: does having trained on a copy of a test story lower that story's
loss? (no training)

  python -B camera_ready/memorization.py     (after contamination.py)

Uses the saved 8-layer E2 seed-0 model (also the E3 base) on the *whole*
TinyStories and Instruct test streams for statistical power. Among test
stories that have an exact copy in the TinyStories training file, whether the
model trained on that copy was decided by the seeded window shuffle, so
exposed-vs-unexposed is a natural experiment -- except that stories with more
copies are more likely to be exposed, so the comparison is also stratified by
copy count (exactly one copy is the clean stratum). Story-level bootstrap.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clean_eval import token_losses  # noqa: E402
from common import BS, OUT, ROOT, ids, mask, save_json  # noqa: E402

sys.path.insert(0, ROOT)
from metrics import load_checkpoint  # noqa: E402

RNG = np.random.default_rng(1)


def per_story(loss, counted, first, nstories):
    pos = np.arange(1, len(loss) + 1)
    owner = np.searchsorted(first, pos, side="right") - 1
    s = np.bincount(owner[counted], weights=loss[counted], minlength=nstories)
    n = np.bincount(owner[counted], minlength=nstories)
    return s, n


def gap(s, n, a, b, boot=2000):
    """token-weighted mean loss of group a minus group b, story bootstrap CI"""
    ia, ib = np.flatnonzero(a & (n > 0)), np.flatnonzero(b & (n > 0))
    est = s[ia].sum() / n[ia].sum() - s[ib].sum() / n[ib].sum()
    bs = []
    for _ in range(boot):
        ka, kb = RNG.choice(ia, len(ia)), RNG.choice(ib, len(ib))
        bs.append(s[ka].sum() / n[ka].sum() - s[kb].sum() / n[kb].sum())
    return {"n_a": int(len(ia)), "n_b": int(len(ib)), "loss_diff": float(est),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main():
    model, tok, device = load_checkpoint(os.path.join(ROOT, "checkpoints", "e2_base_layers8_seed0.pt"))
    res = {"model": "E2 8-layer seed 0 (= E3 base)", "sets": {}}
    for tname, masked in (("tinystories_test", False), ("instruct_test", True)):
        c = dict(np.load(os.path.join(OUT, f"contamination_{tname}.npz")))
        data = np.asarray(ids(tname))
        m = np.asarray(mask(tname)) if masked else None
        loss, counted = token_losses(model, data, m, device)
        s, n = per_story(loss, counted, c["first"], len(c["first"]))
        ex = c["exposure_e2_seed0"]
        dup = c["exact_tinystories_train"]
        k = c["n_copies_tinystories_train"]
        has_garble = np.array(["â€" in tok.decode(data[b:e].tolist()) for b, e in zip(c["body0"], c["eos"])])
        d = {"stories": int(len(n)), "dup_stories": int(dup.sum()),
             "copies_distribution_among_dups": {str(v): int((k[dup] == v).sum()) for v in range(1, 6)} | {">5": int((k[dup] > 5).sum())},
             "all_dups_exposed_ge1_vs_unexposed": gap(s, n, dup & (ex >= 1), dup & (ex == 0)),
             "all_dups_any_exposure_vs_unexposed": gap(s, n, dup & (ex > 0), dup & (ex == 0)),
             "one_copy_exposed_ge1_vs_unexposed": gap(s, n, dup & (k == 1) & (ex >= 1), dup & (k == 1) & (ex == 0)),
             "one_copy_any_exposure_vs_unexposed": gap(s, n, dup & (k == 1) & (ex > 0), dup & (k == 1) & (ex == 0)),
             "multi_copy_vs_one_copy_both_unexposed": gap(s, n, dup & (k > 1) & (ex == 0), dup & (k == 1) & (ex == 0)),
             "garbled_pct": {"dups": float(100 * has_garble[dup].mean()), "non_dups": float(100 * has_garble[~dup].mean()),
                             "non_dup_sharing_50gram": float(100 * has_garble[~dup & (c["ngram50_frac"] > 0)].mean()),
                             "non_dup_no_50gram": float(100 * has_garble[~dup & (c["ngram50_frac"] == 0)].mean())},
             "garbled_vs_clean_loss": gap(s, n, has_garble, ~has_garble)}
        res["sets"][tname] = d
        print(tname, {kk: v for kk, v in d.items()}, flush=True)
    save_json(res, "memorization.json")


if __name__ == "__main__":
    main()
