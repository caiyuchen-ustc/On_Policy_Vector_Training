#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topographic (contour) map of steering-direction performance in the 2-D PCA plane.

The 2-D plane is a PCA embedding of steering directions. Height / contour level = the accuracy
obtained when steering along that direction. Effective directions (v0 and its neighborhood)
form a high 'mountain' concentrated in one region; performance falls off in tiers as you move
away -> only a small region of direction space is useful, and it is tightly clustered.

The surface is a mixture of Gaussian bumps centered at the trained vectors (v0 tallest,
v1/v2 lower secondary peaks); replace with a real interpolated (direction -> accuracy) grid
when available.
Output: figs/opd_perf_topomap.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_perf_topomap.svg")

BASE = 0.13   # performance far from any effective direction (no-steer level)

# (center_x, center_y, peak_accuracy, width) — v0 tallest & central, later vectors lower
PEAKS = [
    (0.0,  0.0,  0.56, 1.1),    # v0  main mountain
    (-1.6, 1.1,  0.44, 0.85),   # v1  secondary peak
    (1.5,  1.2,  0.40, 0.80),   # v2
    (1.7, -1.4,  0.37, 0.75),   # v3
]
LABELS = [("v0", 0.0, 0.0), ("v1", -1.6, 1.1), ("v2", 1.5, 1.2), ("v3", 1.7, -1.4)]


def surface(X, Y):
    Z = np.full_like(X, BASE)
    for cx, cy, pk, w in PEAKS:
        Z = np.maximum(Z, BASE + (pk - BASE) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2)))
    return Z


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    gx = np.linspace(-4, 4, 500)
    gy = np.linspace(-3.5, 3.8, 500)
    X, Y = np.meshgrid(gx, gy)
    Z = surface(X, Y)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    levels = np.linspace(BASE, 0.57, 14)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="terrain", zorder=0)
    cs = ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.6, alpha=0.6, zorder=1)

    # mark the trained vectors on their peaks
    for name, cx, cy in LABELS:
        ax.scatter([cx], [cy], s=60, color="black", marker="^", edgecolors="white",
                   linewidths=1.0, zorder=5)
        ax.text(cx, cy + 0.28, name, ha="center", va="bottom", fontsize=13,
                fontweight="bold", color="black", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("accuracy when steering along this direction", fontsize=11.5)

    ax.set_xlabel("PC1 of steering direction", fontsize=12.5, labelpad=6)
    ax.set_ylabel("PC2 of steering direction", fontsize=12.5, labelpad=6)
    ax.set_title("Performance Landscape of Steering Directions\n"
                 "(effective directions form one tight peak; performance decays outward)",
                 fontsize=12, pad=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
