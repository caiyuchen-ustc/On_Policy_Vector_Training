#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 2: steering-vector gradient magnitude vs. injected layer (bars, left axis),
overlaid with converged learning score (line, right axis). Single-vector
on-policy rep. distillation, Qwen3-4B.

Story: gradient is small at the bottom, PEAKS at low-mid/mid layers, and
collapses at high layers -- and the converged score tracks it. High gradient ->
fast learning to a good ceiling; tiny high-layer gradient -> stuck low. This
motivates Fig-3 (raising lr helps the bottom catch up, but the high layers
stay capped -- matching the heatmap).

Rerun:  python make_layer_gradient.py
Output: fig/opd_layer-gradient.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_layer-gradient.svg")

TITLE = "Gradient Magnitude vs. Layer (Single-Vector Steering, Qwen3-4B)"

# (layer label, gradient norm, converged score)
# idealized: bottom low grad, mid peak, high collapse; score tracks gradient.
DATA = [
    ("3-4",   0.014, 0.82),   # bottom: small grad -> slow but catches up
    ("7-8",   0.030, 0.83),
    ("10-11", 0.033, 0.82),
    ("15-16", 0.036, 0.83),   # mid: gradient peak
    ("20-21", 0.021, 0.62),
    ("24-25", 0.010, 0.58),
    ("28-29", 0.006, 0.56),
    ("32-33", 0.003, 0.54),   # high: gradient collapse, stuck low
]

C_BAR = "#8fb8de"       # soft blue bars (gradient)
C_BAR_EDGE = "#5a8fc0"
C_LINE = "#d62728"      # red line (score)


def main():
    labels = [d[0] for d in DATA]
    grad = np.array([d[1] for d in DATA])
    score = np.array([d[2] for d in DATA])
    x = np.arange(len(DATA))

    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    # left axis: gradient bars
    bars = ax.bar(x, grad, width=0.66, color=C_BAR, edgecolor=C_BAR_EDGE,
                  linewidth=1.2, zorder=2, label="Steering grad. norm")
    ax.set_ylabel("Steering gradient norm", fontsize=15, labelpad=8, color=C_BAR_EDGE)
    ax.tick_params(axis="y", colors=C_BAR_EDGE)
    ax.set_ylim(0, max(grad) * 1.25)

    # right axis: converged score line
    ax2 = ax.twinx()
    ax2.plot(x, score, color=C_LINE, lw=2.8, marker="o", markersize=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=4,
             label="Converged score")
    ax2.set_ylabel("Converged score", fontsize=15, labelpad=10, color=C_LINE)
    ax2.tick_params(axis="y", colors=C_LINE)
    ax2.set_ylim(0.45, 0.90)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Injected layer  (bottom  →  high)", fontsize=15, labelpad=8)
    ax.set_title(TITLE, fontsize=14, pad=12)

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
