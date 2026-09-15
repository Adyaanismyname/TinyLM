"""
Renders the 5 figures referenced by the paper draft in results/report.md,
from the real numbers in results/report_data.json. Output goes to
results/figs/ -- if your LaTeX source lives elsewhere, point its
\\includegraphics paths at results/figs/fig1_loss_vs_params.png etc.
(or copy the folder next to the .tex file).

Run standalone (needs results/report_data.json from run_experiments.py first):

    python3 generate_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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


def fig1_loss_vs_params(scaling):
    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)

    params = [r["params"] for r in scaling]
    losses = [r["val_loss"] for r in scaling]
    labels = ["Small", "Medium", "Large"]

    ax.plot(params, losses, color=BLUE, linewidth=2, marker="o", markersize=6,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)

    for x, y, label in zip(params, losses, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK)

    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation loss")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax.margins(x=0.18, y=0.25)
    save(fig, "fig1_loss_vs_params.png")


def fig2_ppl_vs_params(scaling):
    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)

    params = [r["params"] for r in scaling]
    ppl = [r["val_perplexity"] for r in scaling]
    labels = ["Small", "Medium", "Large"]

    ax.plot(params, ppl, color=BLUE, linewidth=2, marker="o", markersize=6,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)

    for x, y, label in zip(params, ppl, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK)

    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation perplexity")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax.margins(x=0.18, y=0.25)
    save(fig, "fig2_ppl_vs_params.png")


def fig3_pct_trainable_vs_rank(peft_variants):
    lora = [v for v in peft_variants if v["name"].startswith("lora_r")]
    full_ft = next(v for v in peft_variants if v["name"] == "full_finetune")

    ranks = [int(v["name"].split("_r")[1]) for v in lora]
    pct = [v["trainable_pct"] for v in lora]

    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)

    xpos = range(len(ranks))
    bars = ax.bar(xpos, pct, color=BLUE, width=0.55, zorder=3)
    for x, y in zip(xpos, pct):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color=INK)

    ax.axhline(full_ft["trainable_pct"], color=ORANGE, linewidth=1.6, linestyle="--", zorder=2)
    ax.annotate("Full fine-tuning (100%)", (len(ranks) - 1, full_ft["trainable_pct"]),
                textcoords="offset points", xytext=(0, -14), ha="right", fontsize=8, color=ORANGE)

    ax.set_xticks(list(xpos))
    ax.set_xticklabels([f"r={r}" for r in ranks])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Trainable parameters (%)")
    ax.set_ylim(0, 110)
    save(fig, "fig3_pct_trainable_vs_rank.png")


def fig4_rank_vs_ppl(peft_variants):
    lora = [v for v in peft_variants if v["name"].startswith("lora_r")]
    baseline = next(v for v in peft_variants if v["name"] == "baseline_pretrained")
    full_ft = next(v for v in peft_variants if v["name"] == "full_finetune")

    ranks = [int(v["name"].split("_r")[1]) for v in lora]
    ppl = [v["val_perplexity"] for v in lora]

    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)

    ax.plot(ranks, ppl, color=BLUE, linewidth=2, marker="o", markersize=6,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1,
             zorder=3, label="LoRA")

    ax.axhline(baseline["val_perplexity"], color=ORANGE, linewidth=1.6, linestyle="--",
               zorder=2, label="Pretrained baseline")
    ax.axhline(full_ft["val_perplexity"], color=AQUA, linewidth=1.6, linestyle="--",
               zorder=2, label="Full fine-tuning")

    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Validation perplexity")
    ax.set_xticks(ranks)
    ax.margins(y=0.15)
    legend = ax.legend(loc="upper right", fontsize=7, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, "fig4_rank_vs_ppl.png")


def fig5_kvcache_speedup(latency):
    colors = {"small_1M": BLUE, "medium_1.35M": ORANGE, "large_1.7M": AQUA}
    display_names = {"small_1M": "Small (992K)", "medium_1.35M": "Medium (1.30M)", "large_1.7M": "Large (1.73M)"}

    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)
    style_axes(ax)

    ax.axhline(1.0, color=NEUTRAL_REF, linewidth=1.2, linestyle="--", zorder=2)
    ax.annotate("no benefit", (65, 1.0), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=7, color=NEUTRAL_REF)

    for name, rows in latency.items():
        gen_lens = [r["gen_len"] for r in rows]
        speedups = [r["speedup"] for r in rows]
        ax.plot(gen_lens, speedups, color=colors[name], linewidth=2, marker="o",
                 markersize=6, markerfacecolor=colors[name], markeredgecolor=SURFACE,
                 markeredgewidth=1, zorder=3, label=display_names[name])

    ax.set_xlabel("Generated tokens")
    ax.set_ylabel("Speedup (no-cache / cached)")
    ax.set_xticks([20, 50, 100])
    legend = ax.legend(loc="best", fontsize=7, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, "fig5_kvcache_speedup.png")


def main():
    with open(os.path.join(RESULTS_DIR, "report_data.json")) as f:
        data = json.load(f)

    fig1_loss_vs_params(data["scaling"])
    fig2_ppl_vs_params(data["scaling"])
    fig3_pct_trainable_vs_rank(data["peft"]["variants"])
    fig4_rank_vs_ppl(data["peft"]["variants"])
    fig5_kvcache_speedup(data["latency"])


if __name__ == "__main__":
    main()
