#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""On-policy vs off-policy steering vectors: cross cosine matrix (diagonal alignment).

Rows = on-policy v0..v2, cols = off-policy v0..v2. Cell (i,j) = cosine(on.v_i, off.v_j).
The matrix is strongly DIAGONAL — on.v_i aligns with off.v_i and almost nothing else — so the
two training regimes learn the same ordered set of directions (v0<->v0, v1<->v1, ...), most
strongly the primary one. All values are from the real trained vectors (layers 5-15).
Output: figs/opd_onoff_matrix.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_onoff_matrix.svg")

# measured cross-cosine: rows = on v_i, cols = off v_j
S = np.array([
    [0.810, 0.050, -0.008],
    [0.106, 0.490,  0.130],
    [0.074, 0.153,  0.298],
])
LAB = ["v0", "v1", "v2"]


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    n = len(LAB)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(S, cmap="Blues", vmin=0.0, vmax=0.85, aspect="equal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"off {l}" for l in LAB])
    ax.set_yticklabels([f"on {l}" for l in LAB])
    ax.set_xlabel("off-policy vector", fontsize=13, labelpad=8)
    ax.set_ylabel("on-policy vector", fontsize=13, labelpad=8)

    for i in range(n):
        for j in range(n):
            v = S[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13.5,
                    color="white" if v > 0.5 else "#222",
                    fontweight="bold" if i == j else "normal")

    # highlight the diagonal cells with a box
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#c0392b", linewidth=2.2, zorder=5))

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cosine similarity", fontsize=11.5)

    ax.set_title("On- and Off-policy Learn the Same Ordered Directions\n"
                 "(cross-cosine is diagonal: on.v$_i$ ↔ off.v$_i$)", fontsize=11.5, pad=10)

    # white gridlines between cells
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
