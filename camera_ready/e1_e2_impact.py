"""
Task 1c: what the measured overlap effect implies for Experiments 1 and 2,
whose models were not saved (so they cannot be re-evaluated per story).

  python -B camera_ready/e1_e2_impact.py   (after contamination.py and memorization.py)

Inputs:
  * memorization.json -- on the saved 8-layer E2 seed-0 model (whole
    TinyStories test set), the loss difference between duplicated test
    stories whose training copy the model trained on and those it did not,
    with story-bootstrap 95% CIs;
  * contamination_tinystories_test.npz -- each run's exact exposure.
The E1 and E2 models share the 8-layer architecture or the E2 data order:
every E2 model size with the same seed trained on exactly the same windows,
so the only unknown for the scaling fit is how the per-exposure effect
changes with model size; the fit is re-done under explicit assumptions.
"Bias" below = how much lower the reported loss is than it would be with no
overlap, using the 95%-CI bound least favourable to the paper (the largest
plausible memorization benefit), not the point estimate (which is ~0).
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import BS, OUT, ROOT, save_json  # noqa: E402

sys.path.insert(0, ROOT)
import analysis  # noqa: E402

N8 = 1_783_040


def shares(c, limit, key):
    last = ((limit - 1) // BS) * BS
    sel = c["first"] <= last
    own = np.clip(np.minimum(c["eos"], last) - np.maximum(c["first"], 1) + 1, 0, None) * sel
    ex = c[key]
    tot = own.sum()
    return {"ge1_pass": float(own[(ex >= 1)].sum() / tot), "partial": float(own[(ex > 0) & (ex < 1)].sum() / tot)}


def point_fit(sizes, losses, p0):
    popt, _ = curve_fit(lambda n, e, a, al: e + a * np.power(n, -al), sizes, losses,
                        p0=p0, bounds=([0, 0, 0], [np.inf, np.inf, 5]), maxfev=20000)
    return popt


def main():
    mem = json.load(open(os.path.join(OUT, "memorization.json")))["sets"]["tinystories_test"]
    g1 = mem["one_copy_exposed_ge1_vs_unexposed"]     # every TinyStories test duplicate has exactly one copy
    ga = mem["one_copy_any_exposure_vs_unexposed"]
    worst_ge1 = max(0.0, -g1["ci95"][0])   # largest plausible benefit of >=1 full pass (nats/token)
    worst_any = max(0.0, -ga["ci95"][0])   # ... of any (mostly partial) exposure
    c = dict(np.load(os.path.join(OUT, "contamination_tinystories_test.npz")))
    res = {"memorization_8layer": {"ge1_pass_minus_unexposed": g1, "any_exposure_minus_unexposed": ga,
                                   "worst_case_benefit_nats": {"ge1_pass": worst_ge1, "partial": worst_any}}}

    def bias(sh):
        return sh["ge1_pass"] * worst_ge1 + sh["partial"] * worst_any

    e1 = {}
    for pool in (2_000_000, 8_000_000, 32_000_000, 100_000_000):
        rows = []
        for s in (0, 1, 2):
            sh = shares(c, 2_000_000, f"exposure_e1_pool{pool}_seed{s}")
            b = bias(sh)
            rows.append({"seed": s, **sh, "worst_case_loss_bias_nats": b, "worst_case_ppl_bias_pct": 100 * (np.exp(b) - 1)})
        e1[str(pool)] = rows
    res["e1"] = e1

    sh = {s: shares(c, 2_000_000, f"exposure_e2_seed{s}") for s in (0, 1, 2)}
    b8 = float(np.mean([bias(v) for v in sh.values()]))
    res["e2_exposure_shares_by_seed"] = sh
    res["e2_worst_case_bias_8layer_nats"] = b8
    res["e2_worst_case_bias_8layer_ppl_pct"] = 100 * (np.exp(b8) - 1)

    rows, fit = analysis.summarize_e2(os.path.join(ROOT, "results", "runs", "e2"))
    sizes = np.array([r["group"] for r in rows], dtype=float)
    means = np.array([r["mean"] for r in rows])
    p0 = [fit["E"], fit["A"], fit["alpha"]]
    scen = {"reported": {"E": fit["E"], "A": fit["A"], "alpha": fit["alpha"], "alpha_ci95": fit["alpha_ci95"]}}
    # corrected loss = reported + bias_N (removing the benefit), bias_N = b8 * (N/N8)^gamma
    for gamma in (0.0, 1.0, 2.0):
        corr = means + b8 * (sizes / N8) ** gamma
        e, a, al = point_fit(sizes, corr, p0)
        scen[f"bias_scales_as_N^{gamma}"] = {"E": float(e), "A": float(a), "alpha": float(al),
                                             "bias_at_8.8M_nats": float(b8 * (sizes.max() / N8) ** gamma)}
    need = None
    for k in np.concatenate([np.linspace(1.01, 10, 900), np.linspace(10, 2000, 1991)]):
        g = np.log(k) / np.log(sizes.max() / N8)
        e, a, al = point_fit(sizes, means + b8 * (sizes / N8) ** g, p0)
        if abs(al - fit["alpha"]) >= 0.01:
            need = {"multiple_of_8layer_worst_case": float(k), "bias_at_8.8M_nats": float(b8 * k), "alpha_then": float(al)}
            break
    res["e2_fit_scenarios"] = scen
    res["e2_bias_at_8.8M_needed_to_move_alpha_by_0.01"] = need
    save_json(res, "e1_e2_impact.json")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
