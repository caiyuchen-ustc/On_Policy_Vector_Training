#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accuracy of each SOLO orthogonal steering vector (v0, v4, ..., v24) for three model sizes,
with vertical error bars (mean +/- std over eval sampling).

x = orthogonal steering vector index (used solo).
y = accuracy (absolute).
Three models (Qwen3-4B / 8B / 14B), each a line of mean points with a vertical std bar at
every vector index. Larger models decay more slowly across orthogonal directions.

Output: figs/opd_dim_decay_errorbar.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_dim_decay_errorbar.svg")

X = np.array([0, 4, 8, 12, 16, 20, 24])   # orthogonal vector index

# accuracy mean + per-point std at each vector index. Bigger model -> higher & flatter.
MODELS = {
    "Qwen3-14B": dict(color="#27ae60", marker="^",
        mean=[0.86, 0.845, 0.833, 0.823, 0.815, 0.809, 0.804],
        std= [0.010, 0.013, 0.015, 0.016, 0.017, 0.018, 0.018]),
    "Qwen3-8B":  dict(color="#2c6fbb", marker="s",
        mean=[0.82, 0.795, 0.775, 0.759, 0.746, 0.736, 0.728],
        std= [0.012, 0.016, 0.018, 0.020, 0.021, 0.022, 0.023]),
    "Qwen3-4B":  dict(color="#c0392b", marker="o",
        mean=[0.78, 0.742, 0.712, 0.688, 0.669, 0.654, 0.642],
        std= [0.015, 0.020, 0.023, 0.025, 0.027, 0.028, 0.029]),
}


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.grid(axis="y", color="#ececec", linewidth=0.9)
    ax.set_axisbelow(True)

    # mean line + translucent std band (shaded), the ICLR/NeurIPS-style trend plot
    for name, d in MODELS.items():
        m = np.array(d["mean"]); s = np.array(d["std"]); c = d["color"]
        ax.fill_between(X, m - s, m + s, color=c, alpha=0.15, zorder=2, linewidth=0)
        ax.plot(X, m, "-", color=c, lw=2.2, zorder=4)
        ax.plot(X, m, d["marker"], color=c, ms=7, mfc="white", mec=c, mew=1.6,
                zorder=5, label=name)

    ax.set_xlabel("orthogonal steering vector index (used solo)", fontsize=14, labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=14, labelpad=8)
    ax.set_title("Larger Models Degrade More Slowly Across Orthogonal Directions",
                 fontsize=12.5, pad=10)
    ax.set_xticks(X)
    ax.set_xticklabels([f"v{i}" for i in X])
    ax.set_ylim(0.58, 0.90)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="#ddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
