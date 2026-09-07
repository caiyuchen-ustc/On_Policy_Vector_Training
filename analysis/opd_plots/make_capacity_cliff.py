#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant A: capacity-cliff line plot.

x = trainable parameter count on a true log axis, y = converged score, one line
per injection depth. Replaces the categorical 5x5 heatmap: the x axis becomes a
real quantitative variable spanning ~6 orders of magnitude, so the claim
upgrades from "some cells are darker" to "performance is flat across six orders
of magnitude of trainable parameters, then falls off a cliff below a capacity
threshold -- and the cliff only exists at high layers".

Output: fig/opd_capacity-cliff.svg
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_capacity-cliff.svg")

TITLE = "Accuracy Survives Six Orders of Magnitude, Then Falls Off a Cliff"
METRIC = "MATH500 accuracy @ convergence"

FIGSIZE = (7.6, 5.2)
AXES_RECT = {"left": 0.105, "right": 0.775, "top": 0.855, "bottom": 0.155}

# Trainable parameter counts for Qwen3-4B (hidden 2560, 36 layers, |S| = 10
# injected layers). Full-param and the LoRA ranks are derived from the
# architecture; vector steering is |S| x hidden. DERIVED, not measured --
# recompute against the real configs before submission.
#   (label, params, label vertical offset in axes fraction)
METHODS = [
    ("Full-param",      4.02e9, 0.00),
    ("LoRA-16",         6.55e7, 0.07),
    ("LoRA-8",          3.28e7, 0.00),
    ("LoRA-1",          4.10e6, 0.07),
    ("Vector Steering", 2.56e4, 0.00),
]

# Converged score per depth, ordered to match METHODS.
# NOTE: carried over from make_layer_method_heatmap.py GRID. These disagree
# with make_layer_method_scatter.py for the vector column -- reconcile against
# the real sweep before submission.
# Linestyle differentiates otherwise-overlapping lines: the Mid and Bottom rows
# are currently byte-identical in the source GRID, so a solid Mid would be
# completely hidden under Bottom.
DEPTHS = [
    ("High",     [0.84, 0.83, 0.84, 0.83, 0.50], "#b91c1c", "-"),
    ("Mid-High", [0.84, 0.84, 0.83, 0.84, 0.60], "#ea580c", "-"),
    ("Mid",      [0.84, 0.83, 0.84, 0.83, 0.81], "#65a30d", (0, (5, 2))),
    ("Low-Mid",  [0.83, 0.84, 0.83, 0.84, 0.82], "#0d9488", "-"),
    ("Bottom",   [0.84, 0.83, 0.84, 0.83, 0.81], "#1d4ed8", (0, (1.5, 2))),
]

# The flat regime every depth agrees on, drawn as a reference band.
CEILING_LO, CEILING_HI = 0.825, 0.845


def main():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "xtick.labelsize": 12, "ytick.labelsize": 12.5,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = np.array([p for _, p, _ in METHODS])
    ylo, yhi = 0.40, 0.98

    ax.axhspan(CEILING_LO, CEILING_HI, color="#94a3b8", alpha=0.22, zorder=0)
    ax.text(0.985, (CEILING_HI - ylo) / (yhi - ylo) + 0.012, "capacity ceiling",
            transform=ax.transAxes, fontsize=10.5, color="#475569",
            ha="right", va="bottom", style="italic", zorder=1)

    # Thin guide + method name above each measured configuration.
    for (name, p, dy) in METHODS:
        ax.axvline(p, color="#e2e8f0", lw=0.9, zorder=0)
        ax.text(p, 0.885 + dy, name, transform=ax.get_xaxis_transform(),
                fontsize=9.5, color="#475569", ha="center", va="bottom",
                rotation=0, zorder=1)

    for name, scores, color, ls in DEPTHS:
        ax.plot(x, scores, marker="o", ls=ls, color=color, lw=2.2, markersize=7,
                markeredgecolor="white", markeredgewidth=1.4,
                label=name, zorder=3)

    # Annotate the cliff: spread across depths at the smallest budget.
    tail = [s[-1] for _, s, _, _ in DEPTHS]
    ax.annotate("", xy=(x[-1], max(tail)), xytext=(x[-1], min(tail)),
                arrowprops=dict(arrowstyle="<->", color="#334155", lw=1.4),
                zorder=4)
    # Park the label in the empty lower-right corner: right of x[-1] it hits the
    # legend, and between the arrow tips it sits on top of the Mid-High line.
    ax.text(x[-1] * 1.9, 0.452, "depth matters\nonly here",
            fontsize=10.5, color="#334155",
            ha="center", va="center", linespacing=1.35, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(1.6e10, 1.2e4)   # many params on the left, few on the right
    ax.set_ylim(ylo, yhi)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Trainable parameters  (many  →  few)", fontsize=15, labelpad=8)
    ax.set_ylabel(METRIC, fontsize=13.5, labelpad=8)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.set_title(TITLE, fontsize=13.5, pad=14)
    ax.legend(title="Injection depth", loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=10.5, title_fontsize=11, frameon=True, framealpha=0.95,
              edgecolor="#dddddd", labelspacing=0.6)

    fig.subplots_adjust(**AXES_RECT)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
