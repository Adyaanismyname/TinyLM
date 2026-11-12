"""
Renders the paper's figures from the per-run JSON files under
results/runs/{e1,e2,e3,e4,e5}/, via analysis.py's aggregation (mean + 95% CI
over seeds, and the Experiment 2 power-law fit). Every figure plots the
individual per-seed points alongside the mean, so a reader can see the
spread a single-number summary would hide -- the paper's original version
had no seeds and so no spread to show.

Run (needs at least one experiment's results/runs/<name>/ populated by
run_experiments.py / bench_train.py / bench_kvcache.py first):

    python3 generate_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from analysis import load_runs, summarize_e1, summarize_e2, summarize_e3

# ---- palette (fixed categorical order: Small/LoRA -> blue, Medium/baseline
# -> orange, Large/full-finetune -> aqua; identity stays consistent across
# every figure, never reassigned by rank or value) -----------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
NEUTRAL_REF = "#898781"

RESULTS_DIR = "results"
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIGS_DIR = os.path.join(RESULTS_DIR, "figs")
DPI = 300
SINGLE_COL_SIZE = (3.4, 2.55)  # inches, fits an IEEE single column


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)


def save(fig, name):
    os.makedirs(FIGS_DIR, exist_ok=True)
    path = os.path.join(FIGS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def plot_mean_with_points(ax, rows, color, jitter_frac=0.02):
    """Plots aggregate.py-style rows (group/mean/ci95/values) as a mean
    line with error bars, plus each seed's individual value as a small
    jittered dot underneath -- so a reader sees both the summary and the
    spread it's summarizing."""
    xs = [r["group"] for r in rows]
    means = [r["mean"] for r in rows]
    errs = [r["ci95"] for r in rows]

    ax.errorbar(xs, means, yerr=errs, color=color, linewidth=2, marker="o", markersize=6,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1,
                capsize=3, zorder=3)

    rng = np.random.default_rng(0)
    for x, row in zip(xs, rows):
        if row["n"] <= 1:
            continue
        jitter = rng.uniform(-1, 1, size=row["n"]) * x * jitter_frac
        ax.scatter([x] * row["n"] + jitter, row["values"], color=color, alpha=0.35, s=14, zorder=2)


def fig_e1_data_scaling():
    rows = summarize_e1(os.path.join(RUNS_DIR, "e1"))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)
    plot_mean_with_points(ax, rows, BLUE)
    ax.set_xscale("log")
    ax.set_xlabel("Unique training tokens")
    ax.set_ylabel("Validation perplexity")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    save(fig, "fig_e1_data_scaling.png")


def fig_e2_scaling_law():
    rows, fit = summarize_e2(os.path.join(RUNS_DIR, "e2"))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)
    plot_mean_with_points(ax, rows, BLUE)

    if fit is not None:
        xs = np.geomspace(min(r["group"] for r in rows), max(r["group"] for r in rows), 200)
        ax.plot(xs, fit["fit_fn"](xs), color=ORANGE, linewidth=1.4, linestyle="--", zorder=2,
                label=f"fit: L = {fit['E']:.2f} + {fit['A']:.1f}·N^-{fit['alpha']:.2f}")
        legend = ax.legend(loc="upper right", fontsize=6.5, frameon=False)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation loss")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.2g}M"))
    save(fig, "fig_e2_scaling_law.png")


def fig_e3_lora_vs_fullft():
    rows = summarize_e3(os.path.join(RUNS_DIR, "e3"))
    if not rows:
        return
    ppl_rows = [r for r in rows if r["metric"] == "instruct_val_perplexity"]
    if not ppl_rows:
        return

    lora_rows = sorted([r for r in ppl_rows if r["arm"].startswith("lora_r")],
                        key=lambda r: r["trainable_params"])
    baseline = next((r for r in ppl_rows if r["arm"] == "baseline_frozen"), None)
    full_ft = next((r for r in ppl_rows if r["arm"] == "full_finetune"), None)

    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)

    if lora_rows:
        xs = [r["trainable_params"] for r in lora_rows]
        means = [r["mean"] for r in lora_rows]
        errs = [r["ci95"] for r in lora_rows]
        ax.errorbar(xs, means, yerr=errs, color=BLUE, linewidth=2, marker="o", markersize=6,
                    markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1,
                    capsize=3, zorder=3, label="LoRA")

    if baseline is not None:
        ax.axhline(baseline["mean"], color=ORANGE, linewidth=1.6, linestyle="--", zorder=2,
                   label="No fine-tuning")
    if full_ft is not None:
        ax.axhline(full_ft["mean"], color=AQUA, linewidth=1.6, linestyle="--", zorder=2,
                   label=f"Full fine-tuning ({full_ft['trainable_params']:,} params)")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Instruct validation perplexity")
    legend = ax.legend(loc="best", fontsize=6.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, "fig_e3_lora_vs_fullft.png")


def fig_e3_forgetting():
    rows = summarize_e3(os.path.join(RUNS_DIR, "e3"))
    if not rows:
        return
    forget_rows = sorted(
        [r for r in rows if r["metric"] == "tinystories_forget_val_perplexity"],
        key=lambda r: r["trainable_params"],
    )
    if not forget_rows:
        return

    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)
    xs = [r["trainable_params"] for r in forget_rows]
    means = [r["mean"] for r in forget_rows]
    errs = [r["ci95"] for r in forget_rows]
    labels = [r["arm"] for r in forget_rows]

    xpos = range(len(xs))
    ax.errorbar(xpos, means, yerr=errs, color=BLUE, linewidth=0, marker="o", markersize=7,
                markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1, capsize=3, zorder=3)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("TinyStories perplexity\n(forgetting check)")
    save(fig, "fig_e3_forgetting.png")


def fig_e4_train_speed():
    import glob
    import json

    paths = glob.glob(os.path.join(RUNS_DIR, "e4", "*.json"))
    if not paths:
        return

    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)
    colors = {"full": AQUA, "lora_r8": BLUE, "lora_r64": ORANGE}

    for path in paths:
        with open(path) as f:
            data = json.load(f)
        device = {
            "NVIDIA GeForce RTX 3050 6GB Laptop GPU": "RTX 3050 Laptop",
        }.get(data["device"], data["device"])
        by_method = {}
        for r in data["results"]:
            by_method.setdefault(r["method"], []).append((r["width"], r["summary"]["p50"]))
        for method, points in by_method.items():
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=colors.get(method, INK_MUTED), linewidth=1.6, marker="o",
                    markersize=4, label=f"{method} ({device})", alpha=0.85)

    ax.set_xlabel("Model width (embed_dim)")
    ax.set_ylabel("Median step time (s)")
    ax.set_xscale("log", base=2)
    legend = ax.legend(loc="upper left", fontsize=5.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, "fig_e4_train_speed.png")


def fig_e5_kvcache_speedup():
    import glob
    import json

    paths = glob.glob(os.path.join(RUNS_DIR, "e5", "*.json"))
    if not paths:
        return

    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)
    ax.axhline(1.0, color=NEUTRAL_REF, linewidth=1.2, linestyle="--", zorder=2)

    colors = {"concat": BLUE, "prealloc": ORANGE}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        by_key = {}
        none_by_len_batch = {}
        for r in data["results"]:
            key = (r["batch_size"], r["gen_len"])
            if r["cache_mode"] == "none":
                none_by_len_batch[key] = r["seconds"]["p50"]
            else:
                by_key.setdefault((r["cache_mode"], r["batch_size"]), []).append((r["gen_len"], r["seconds"]["p50"]))

        for (cache_mode, batch_size), points in by_key.items():
            points.sort()
            xs, ys = [], []
            for gen_len, cached_time in points:
                base = none_by_len_batch.get((batch_size, gen_len))
                if base is not None:
                    xs.append(gen_len)
                    ys.append(base / cached_time)
            if xs:
                linestyle = "-" if batch_size == 16 else ":"
                ax.plot(xs, ys, color=colors.get(cache_mode, INK_MUTED), linewidth=1.6, marker="o",
                        markersize=4, linestyle=linestyle,
                        label=f"{cache_mode} (batch={batch_size})", alpha=0.85)

    ax.set_xlabel("Generated tokens")
    ax.set_ylabel("Speedup (no-cache / cached)")
    legend = ax.legend(loc="best", fontsize=6, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, "fig_e5_kvcache_speedup.png")


def main():
    fig_e1_data_scaling()
    fig_e2_scaling_law()
    fig_e3_lora_vs_fullft()
    fig_e3_forgetting()
    fig_e4_train_speed()
    fig_e5_kvcache_speedup()


if __name__ == "__main__":
    main()
