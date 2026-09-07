#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heatmap of the depth x parameterization sweep, with cells encoding the DELTA
against the Sub-Full FT baseline at the same depth instead of the raw score.
"Sub-Full FT" = full-parameter fine-tuning, but restricted to the sublayers in
that depth band -- every column in this figure trains only that subset, so
calling the baseline "Full-param" would overstate what it touches.

Rationale: in the raw-score version (make_layer_method_heatmap.py) 20 of 25
cells sit at 0.83-0.84, so 80% of the ink says "nothing happened" while the
colour scale still assigns those cells visibly different shades -- inviting
readers to over-read 0.83 vs 0.84 noise. Encoding the delta collapses that
block to a near-zero neutral field and lets the one real effect (the
vector-steering cliff at high layers) carry all the colour.

Figure proportions follow make_layer_method_heatmap.py so the two are
interchangeable in the paper.

Output: fig/opd_layer-method-delta.svg
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_layer-method-delta.svg")

TITLE = "Accuracy Change Relative to Full-param training"
SUBTITLE = ("first column: absolute MATH500 accuracy\n"
            "other columns: difference from it at the same depth")

COLS = ["Full-param", "LoRA-16", "LoRA-8", "LoRA-4", "LoRA-1", "Vector Steering"]
ROWS = ["High", "Mid-High", "Mid", "Low-Mid", "Bottom"]

# Depth bands: 36 layers (Qwen3-4B) split into five equal slices. These bounds
# were previously nowhere in the repo -- only the five labels were -- which is why
# the vector score at "high layers" could disagree between figures without anyone
# being able to say whether 25-28 meant Mid-High or High. Anything quoting a band
# must resolve it here.
BANDS = {
    "High":     (29, 35),
    "Mid-High": (22, 28),
    "Mid":      (14, 21),
    "Low-Mid":  (7, 13),
    "Bottom":   (0, 6),
}

# Raw converged scores, carried over from make_layer_method_heatmap.py.
# NOTE: these disagree with make_layer_method_scatter.py for the vector column.
#
# Mid-High vector = 0.64, not the 0.60 this table used to carry. 0.64 is the value
# that appears independently in make_gate_ablations.py (the capacity ladder's
# starting point, LAYER = "high layer 25-28") and in make_gated_highlayers_multi.py
# ("Layer 25-28", plain = 0.64), i.e. twice, from the gate experiments -- whereas
# 0.60 appeared only here. Layers 25-28 fall inside BANDS["Mid-High"] = (22, 28),
# so those figures and this cell describe the same measurement and must agree:
# section 2 puts them side by side and the ladder's whole story is "climb from the
# failure point", so a 0.60/0.64 mismatch would be read off the page directly.
GRID = {
    #            Full  LoRA16 LoRA8 LoRA4 LoRA1 Vector
    "High":     [0.84, 0.83, 0.84, 0.83, 0.83, 0.50],
    "Mid-High": [0.84, 0.84, 0.83, 0.84, 0.84, 0.64],
    "Mid":      [0.84, 0.83, 0.84, 0.84, 0.83, 0.81],
    "Low-Mid":  [0.83, 0.84, 0.83, 0.83, 0.84, 0.82],
    "Bottom":   [0.84, 0.83, 0.84, 0.83, 0.83, 0.81],
}

# Diverging: loss = red, parity = white, gain = teal. TwoSlopeNorm maps
# vcenter to 0.5 of the colormap, so white must sit exactly at the midpoint of
# the colour list -- otherwise a Delta of 0.00 renders pink and the "no effect"
# block reads as a small loss.
CMAP = LinearSegmentedColormap.from_list(
    "delta", ["#7f1d1d", "#dc2626", "#fca5a5", "#ffffff", "#99f6e4", "#14b8a6", "#0f766e"]
)
NORM = TwoSlopeNorm(vmin=-0.36, vcenter=0.0, vmax=0.36)


def main():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#888888", "axes.linewidth": 1.0,
        "xtick.labelsize": 11, "ytick.labelsize": 12.5,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    # Narrow cells: 6 columns in ~5.1in of grid plus ~0.6in for the colorbar.
    # The cells are deliberately taller than wide so the 6 methods read as a
    # sequence along x while the depth bands stay the dominant vertical axis.
    fig, ax = plt.subplots(figsize=(5.8, 6.4))
    ncol, nrow = len(COLS), len(ROWS)

    for i, row in enumerate(ROWS):
        base = GRID[row][0]
        for j, perf in enumerate(GRID[row]):
            d = perf - base
            # The baseline column is Delta=0 by construction, so colouring it
            # would be meaningless (pure white = a hole in the grid). Give it a
            # grey wash and print the ABSOLUTE accuracy instead: that is the
            # anchor every other cell is measured against, and it is the one
            # number a reader needs to convert a Delta back to a score.
            face = "#dbe3ec" if j == 0 else CMAP(NORM(d))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face,
                                       edgecolor="white", lw=1.8, zorder=2))
            strong = abs(d) > 0.22
            label = f"{perf:.2f}" if j == 0 else f"{d:+.2f}"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
                    fontsize=10, zorder=3,
                    fontweight="bold" if j == 0 else "medium",
                    color="white" if strong else "#334155")

    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_xticks(np.arange(ncol) + 0.5)
    # 16 deg (as in make_layer_method_heatmap.py) is too shallow here:
    # "LoRA-1" and "Vector Steering" collide.
    ax.set_xticklabels(COLS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(nrow) + 0.5)
    ax.set_yticklabels(ROWS)
    ax.invert_yaxis()   # High at top
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(TITLE, fontsize=12.5, pad=40)
    ax.text(0.5, 1.012, SUBTITLE, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.5, color="#64748b", style="italic",
            linespacing=1.5)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.03)
    cb.set_label("Δ accuracy vs. Full-param", fontsize=11, labelpad=8)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
