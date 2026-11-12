"""
Aggregates the per-run JSON files run_experiments.py / bench_train.py /
bench_kvcache.py write under results/runs/ into the summary tables and
figures the paper actually cites. Nothing here re-runs any training --
every number is read from a run's saved final_metrics, never recomputed or
hand-typed, so the report can't silently drift from what the runs actually
measured.

    from analysis import load_runs, aggregate, fit_power_law
"""

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def load_runs(directory):
    """Loads every *.json file directly under `directory` (non-recursive)
    and returns them as a list of dicts (train.save_run_json's schema:
    config/seed/device/history/final_metrics)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            run = json.load(f)
        run["_path"] = path
        runs.append(run)
    return runs


def mean_ci95(values):
    """Mean and a 95% confidence half-width using the t-distribution (not
    the normal approximation), since these experiments deliberately use
    small seed counts (3-5) where the t-distribution's fatter tails matter.
    With n=1, returns a half-width of 0.0 (nothing to estimate spread from)
    rather than raising or returning NaN, so a group that's only been run
    once still plots as a point rather than breaking the aggregation."""
    values = list(values)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    sem = float(np.std(values, ddof=1)) / math.sqrt(n)
    half_width = sem * stats.t.ppf(0.975, df=n - 1)
    return mean, half_width


def aggregate(runs, group_key, value_key):
    """
    Groups `runs` (as returned by load_runs) by `group_key(run)` and
    computes mean + 95% CI of `value_key(run)` (usually a
    run["final_metrics"][...] lookup) within each group -- e.g. one point
    per model size in Experiment 2, averaged over that size's seeds.

    Returns a list of {"group": key, "n": count, "mean": ..., "ci95": ...,
    "values": [...]} dicts, sorted by group key, so plotting code can just
    read x/y/error directly off it without repeating the statistics.
    """
    groups = defaultdict(list)
    for run in runs:
        groups[group_key(run)].append(value_key(run))

    rows = []
    for key in sorted(groups, key=lambda k: (isinstance(k, str), k)):
        values = groups[key]
        mean, ci95 = mean_ci95(values)
        rows.append({"group": key, "n": len(values), "mean": mean, "ci95": ci95, "values": values})
    return rows


def fit_power_law(sizes, losses, num_bootstrap=2000, seed=0):
    """
    Fits L(N) = E + A * N^-alpha (the standard neural scaling-law form) to
    (sizes, losses) via nonlinear least squares, and estimates a 95%
    confidence interval for each parameter by bootstrap resampling the
    (size, loss) pairs with replacement and refitting -- appropriate here
    since Experiment 2 has few, unevenly-spaced points rather than enough
    data for the fit's own asymptotic covariance estimate to be trustworthy.

    Returns {"E":, "A":, "alpha":, "E_ci95":, "A_ci95":, "alpha_ci95":,
    "fit_fn": callable}, or None if the fit doesn't converge (e.g. too few
    distinct sizes) -- callers should handle that rather than assume a fit
    always succeeds so a bad sweep doesn't crash figure generation.
    """
    from scipy.optimize import curve_fit

    sizes = np.asarray(sizes, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if len(set(sizes.tolist())) < 3:
        return None

    def model(n, e, a, alpha):
        return e + a * np.power(n, -alpha)

    p0 = [max(losses.min() - 0.1, 0.0), 1.0, 0.3]
    bounds = ([0, 0, 0], [np.inf, np.inf, 5])

    try:
        popt, _ = curve_fit(model, sizes, losses, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        return None

    rng = np.random.default_rng(seed)
    boot_params = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, len(sizes), size=len(sizes))
        try:
            p, _ = curve_fit(model, sizes[idx], losses[idx], p0=popt, bounds=bounds, maxfev=20000)
            boot_params.append(p)
        except RuntimeError:
            continue

    result = {"E": float(popt[0]), "A": float(popt[1]), "alpha": float(popt[2])}
    if boot_params:
        boot_params = np.array(boot_params)
        for i, name in enumerate(["E", "A", "alpha"]):
            lo, hi = np.percentile(boot_params[:, i], [2.5, 97.5])
            result[f"{name}_ci95"] = (float(lo), float(hi))
    else:
        for name in ["E", "A", "alpha"]:
            result[f"{name}_ci95"] = (result[name], result[name])

    result["fit_fn"] = lambda n: model(np.asarray(n, dtype=np.float64), *popt)
    return result


def summarize_e1(runs_dir="results/runs/e1"):
    """One row per unique_data_tokens value, averaged over seeds."""
    runs = load_runs(runs_dir)
    return aggregate(
        runs,
        group_key=lambda r: r["config"]["unique_data_tokens"],
        value_key=lambda r: r["final_metrics"]["val_perplexity"],
    )


def summarize_e2(runs_dir="results/runs/e2"):
    """One row per model size (params), averaged over seeds, plus a fitted
    power law over the group means."""
    runs = load_runs(runs_dir)
    rows = aggregate(
        runs,
        group_key=lambda r: r["final_metrics"]["params"],
        value_key=lambda r: r["final_metrics"]["val_loss"],
    )
    fit = fit_power_law([r["group"] for r in rows], [r["mean"] for r in rows]) if len(rows) >= 3 else None
    return rows, fit


def summarize_e3(runs_dir="results/runs/e3"):
    """One row per arm (baseline_frozen, lora_r*, full_finetune), averaged
    over seeds, keyed by trainable-parameter count so it plots directly
    against LoRA rank / full fine-tuning on a shared x-axis."""
    runs = load_runs(runs_dir)
    arm_of = lambda r: r["_path"].split(os.sep)[-1].rsplit("_seed", 1)[0]
    groups = defaultdict(list)
    for run in runs:
        groups[arm_of(run)].append(run)

    rows = []
    for arm, arm_runs in sorted(groups.items()):
        trainable = arm_runs[0]["final_metrics"]["trainable_params"]
        for metric in ["instruct_val_perplexity", "tinystories_forget_val_perplexity", "words_constraint_rate"]:
            mean, ci95 = mean_ci95([r["final_metrics"][metric] for r in arm_runs])
            rows.append({"arm": arm, "trainable_params": trainable, "metric": metric,
                         "n": len(arm_runs), "mean": mean, "ci95": ci95})
    return rows


if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "results/runs"
    for name, fn in [("e1", summarize_e1), ("e3", summarize_e3)]:
        directory = os.path.join(base, name)
        if os.path.isdir(directory) and glob.glob(os.path.join(directory, "*.json")):
            print(f"\n=== {name} ===")
            for row in fn(directory):
                print(row)

    e2_dir = os.path.join(base, "e2")
    if os.path.isdir(e2_dir) and glob.glob(os.path.join(e2_dir, "*.json")):
        print("\n=== e2 ===")
        rows, fit = summarize_e2(e2_dir)
        for row in rows:
            print(row)
        print("power law fit:", {k: v for k, v in (fit or {}).items() if k != "fit_fn"})
