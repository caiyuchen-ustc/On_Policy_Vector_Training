#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer (y) vs. trainable-parameter count (x) scatter; marker SIZE = converged
performance (RL-score recovery). Color = method.

Claim visualized: only Vector-steering injected at HIGH layers fails (tiny dot),
because an additive constant vector has no input-dependent non-linearity and
must borrow the (few remaining) downstream non-linear blocks. LoRA / Full carry
their own input-dependent capacity, so they work at every depth. => "RL can be
contained in very few parameters, but only above a minimum non-linear capacity;
not *any* few / *any* position."

Edit CONFIG (method, params, layer, perf) and rerun:
    python make_layer_param_scatter.py
Output: fig/opd_layer-vs-param.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_layer-vs-param.svg")

TITLE = "Where RL Can Be Contained: Layer vs. Trainable Parameters"
# y-axis layer bands (row index -> label). Higher row = closer to output.
LAYER_LABELS = ["Low\n(4-8)", "Mid\n(12-16)", "High\n(20-24)", "Top\n(28-32)"]

METHOD_COLOR = {"Vector Steering": "#2ca02c", "LoRA": "#d62728", "Full-param": "#1f77b4"}

# marker size = perf: area scales with perf so the eye reads "big dot = good".
SIZE_MIN, SIZE_MAX = 40, 900     # marker area (pt^2) for perf 0..1
PERF_LO, PERF_HI = 0.15, 0.90    # perf range mapped onto [SIZE_MIN, SIZE_MAX]

# ----------------------------------------------------------------------------
# Data: (method, trainable_params, layer_row_idx, converged_perf 0..1)
#   trainable_params in absolute count (log x-axis). layer_row_idx indexes
#   LAYER_LABELS (0=Low ... 3=Top). perf = RL-score recovery at convergence.
#   >>> replace the numbers with your real sweep <<<
# ----------------------------------------------------------------------------
D_MODEL = 4096          # a steering vector ~ d params
CONFIG = [
    # Vector steering: params ~ D_MODEL at every layer; FAILS only at Top.
    ("Vector Steering", D_MODEL,        0, 0.86),
    ("Vector Steering", D_MODEL,        1, 0.84),
    ("Vector Steering", D_MODEL,        2, 0.55),
    ("Vector Steering", D_MODEL,        3, 0.22),   # high layer -> tiny dot (fails)
    # LoRA rank sweep (params grow with rank); works at all depths.
    ("LoRA", 2 * 8  * D_MODEL,          0, 0.87),
    ("LoRA", 2 * 8  * D_MODEL,          3, 0.83),
    ("LoRA", 2 * 32 * D_MODEL,          0, 0.88),
    ("LoRA", 2 * 32 * D_MODEL,          3, 0.86),
    # Full-param: largest params; works everywhere.
    ("Full-param", 200 * 10**6,         0, 0.90),
    ("Full-param", 200 * 10**6,         3, 0.89),
]


def _perf_to_size(p):
    frac = np.clip((p - PERF_LO) / (PERF_HI - PERF_LO), 0.0, 1.0)
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * frac


def main():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#444444", "axes.linewidth": 1.1,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.8,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.5))

    # jitter overlapping same-(x,y) points from different methods a touch in y
    seen = {}
    for method, params, row, perf in CONFIG:
        key = (round(np.log10(params), 3), row)
        k = seen.get(key, 0); seen[key] = k + 1
        y = row + (k * 0.12)  # small vertical offset if collisions
        ax.scatter(params, y, s=_perf_to_size(perf), c=METHOD_COLOR[method],
                   alpha=0.75, edgecolors="white", linewidths=1.3, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters", fontsize=17, labelpad=8)
    ax.set_ylabel("Injected / trained layer", fontsize=17, labelpad=8)
    ax.set_title(TITLE, fontsize=15, pad=12)
    ax.set_yticks(range(len(LAYER_LABELS)))
    ax.set_yticklabels(LAYER_LABELS)
    ax.set_ylim(-0.6, len(LAYER_LABELS) - 0.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # legend 1: method colors
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], marker="o", ls="", markersize=11,
                            markerfacecolor=c, markeredgecolor="white", label=m)
                     for m, c in METHOD_COLOR.items()]
    leg1 = ax.legend(handles=color_handles, loc="lower right", fontsize=11,
                     framealpha=0.9, edgecolor="#cccccc", title="Method",
                     title_fontsize=11)
    ax.add_artist(leg1)

    # legend 2: size = performance
    size_vals = [0.25, 0.55, 0.85]
    size_handles = [Line2D([0], [0], marker="o", ls="", markeredgecolor="#555",
                           markerfacecolor="#bbbbbb",
                           markersize=np.sqrt(_perf_to_size(p)),
                           label=f"{p:.2f}") for p in size_vals]
    ax.legend(handles=size_handles, loc="upper left", fontsize=10,
              framealpha=0.9, edgecolor="#cccccc", title="Converged score",
              title_fontsize=11, labelspacing=1.4, borderpad=1.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
