#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5x5 cosine-similarity heatmap of v0 across four SciKnowEval domains + the mixed-trained run.

Companion to the teacher-invariance and regime-invariance figures. Those fix the task and vary
the training choice -> the primary direction is unchanged. Here we VARY the task (domain):
  - the four domains' v0 are only 0.27-0.43 apart -> different tasks have different primary
    directions (the direction is a property of the task, so changing the task changes it);
  - the mixed-trained v0 aligns with each domain in proportion to its data (chemistry 0.75,
    largest domain; physics 0.51, etc.) -> mixed training is a data-weighted combination of the
    per-domain directions, not a new one.
All values are from the real trained vectors (step-100 v0, layers 5-15).
Output: figs/opd_domain_mix_heatmap.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_domain_mix_heatmap.svg")

LABELS = ["Physics", "Chemistry", "Biology", "Material", "Mixed"]
S = np.array([
    [1.000, 0.428, 0.366, 0.380, 0.508],
    [0.428, 1.000, 0.374, 0.314, 0.747],
    [0.366, 0.374, 1.000, 0.267, 0.537],
    [0.380, 0.314, 0.267, 1.000, 0.570],
    [0.508, 0.747, 0.537, 0.570, 1.000],
])


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    n = len(LABELS)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(S, cmap="YlGnBu", vmin=0.2, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, rotation=25, ha="right")
    ax.set_yticklabels(LABELS)

    for i in range(n):
        for j in range(n):
            v = S[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                    color="white" if v > 0.62 else "#222")

    # separator line marking the "Mixed" row/col
    ax.axhline(3.5, color="#c0392b", lw=2.0)
    ax.axvline(3.5, color="#c0392b", lw=2.0)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cosine similarity of v0", fontsize=12)

    ax.set_title("Domains Differ; Mixed Training Fuses Them by Data",
                 fontsize=13, pad=10)
    # thin white gridlines
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
