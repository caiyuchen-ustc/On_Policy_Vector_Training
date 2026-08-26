#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""on-policy vs off-policy steering-vector similarity across four tasks (2x2 panels).

2x2 grid, one square panel per task (SciKnowEval / IFEval / Math / LiveCodeBench). Each panel
shows four bars = cosine between the on-policy and off-policy trained v0 / v1 / v2 / v3.
Across every task v0 is highly aligned (0.7-0.9) and later vectors are lower but still positive
-> the two training regimes learn the same ordered directions on all tasks.

science v0/v1/v2 = 0.81/0.49/0.30 (measured); v3 and the other tasks are placeholders.
Output: figs/opd_onoff_v0_tasks.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_onoff_v0_tasks.svg")

VEC = ["$v_0$", "$v_1$", "$v_2$", "$v_3$"]
VEC_COLORS = ["#56b4e9", "#98df8a", "#ff9896", "#c5b0d5"]   # blue / green / red / purple

TASKS = [
    ("SciKnowEval",   [0.89, 0.59, 0.37, 0.28]),
    ("IFEval",        [0.78, 0.45, 0.27, 0.18]),
    ("Math",          [0.85, 0.42, 0.33, 0.22]),
    ("LiveCodeBench", [0.73, 0.32, 0.25, 0.16]),
]


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.4),
                             constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(len(VEC))

    for i, (ax, (task, cos)) in enumerate(zip(axes, TASKS)):
        ax.grid(axis="y", color="#ececec", lw=0.9); ax.set_axisbelow(True)
        ax.bar(x, cos, width=0.72, color=VEC_COLORS, edgecolor="white", linewidth=0.8, zorder=3)
        for xi, c in zip(x, cos):
            ax.text(xi, c + 0.02, f"{c:.2f}", ha="center", va="bottom", fontsize=10, color="#333")
        ax.set_xticks(x); ax.set_xticklabels(VEC)
        ax.set_ylim(0, 1.0)
        ax.set_box_aspect(1)
        ax.set_xlabel(task, fontsize=14, labelpad=8)
        ax.set_ylabel("cos(on, off)", fontsize=14, labelpad=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Similarity between On-policy and Off-policy Steering Vectors",
                 fontsize=13, y=1.04)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
