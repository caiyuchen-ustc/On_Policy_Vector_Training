#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heatmap: method (x) vs. injected/trained depth (y). Cell shade = converged
performance, with WORSE = DARKER (reversed) on an overall light palette.

Claim: only Vector Steering at HIGH layers degrades (dark cell); LoRA / Full
stay light (good) at every depth they were tried. => RL can be contained in
very few params, but only above a minimum non-linear capacity.

Edit GRID and rerun:
    python make_layer_method_heatmap.py
Output: fig/opd_layer-method-heatmap.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_layer-method-heatmap.svg")

TITLE = "Distillation Performance versus Model Depth and Parameter Scale"

# columns (left->right): many params -> few params
COLS = ["Full-param", "LoRA-16", "LoRA-8", "LoRA-1", "Vector Steering"]
# rows top->bottom (High at top)
ROWS = ["High", "Mid-High", "Mid", "Low-Mid", "Bottom"]

# perf value per (row, col). Every cell filled.
# Full-param / LoRA-* are all ~0.83-0.84 (similar, good at every depth).
# Vector Steering: bottom/mid ~0.81-0.82, then drops off at the high layers.
GRID = {
    #            Full  LoRA16 LoRA8 LoRA1  Vector
    "High":     [0.84, 0.83, 0.84, 0.83, 0.50],
    "Mid-High": [0.84, 0.84, 0.83, 0.84, 0.60],
    "Mid":      [0.84, 0.83, 0.84, 0.83, 0.81],
    "Low-Mid":  [0.83, 0.84, 0.83, 0.84, 0.82],
    "Bottom":   [0.84, 0.83, 0.84, 0.83, 0.81],
}

# perf -> shade. worse (low perf) = DARKER; good = light. overall light palette.
PERF_LO, PERF_HI = 0.15, 0.92
# light -> deep, but not fully saturated (keep it airy)
CMAP = LinearSegmentedColormap.from_list(
    "worse_darker", ["#f2faf6", "#c7ebd7", "#7fcea8", "#2f9e68", "#12633c"]
)


def _shade(perf):
    # frac: 1 when worst (dark), 0 when best (light)
    frac = 1.0 - np.clip((perf - PERF_LO) / (PERF_HI - PERF_LO), 0.0, 1.0)
    return CMAP(frac)


def main():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#888888", "axes.linewidth": 1.0,
        "xtick.labelsize": 12.5, "ytick.labelsize": 13.5,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(5.7, 6.4))
    ncol, nrow = len(COLS), len(ROWS)

    for i, row in enumerate(ROWS):
        for j, perf in enumerate(GRID[row]):
            if perf is None:
                # blank / not-measured cell: faint hatch-free light grey
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#fbfbfb",
                                           edgecolor="#eeeeee", lw=1.0, zorder=1))
                continue
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=_shade(perf),
                                       edgecolor="white", lw=2.2, zorder=2))
            # annotate value; text color adapts to cell darkness
            frac = 1.0 - np.clip((perf - PERF_LO) / (PERF_HI - PERF_LO), 0.0, 1.0)
            txtc = "white" if frac > 0.62 else "#124a30"
            ax.text(j + 0.5, i + 0.5, f"{perf:.2f}", ha="center", va="center",
                    fontsize=12.5, color=txtc, zorder=3, fontweight="medium")

    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow)
    ax.set_xticks(np.arange(ncol) + 0.5); ax.set_xticklabels(COLS, rotation=16, ha="right")
    ax.set_yticks(np.arange(nrow) + 0.5); ax.set_yticklabels(ROWS)
    ax.invert_yaxis()   # High at top
    # ax.set_xlabel("Parameterization  (many  →  few parameters)", fontsize=15, labelpad=8)
    # ax.set_ylabel("Injection depth", fontsize=16, labelpad=8)
    ax.set_title(TITLE, fontsize=13.5, pad=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
