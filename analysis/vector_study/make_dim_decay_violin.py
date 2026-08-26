#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Violin (grouped) version: at each orthogonal vector index (v0, v4, ..., v24), the three
models each show a violin of the accuracy DISTRIBUTION (over eval samples/seeds), not just
mean+/-std. Larger models keep higher, tighter distributions as later vectors are used.

Samples are drawn from each point's (mean, std) to render the distribution.
Output: figs/opd_dim_decay_violin.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_dim_decay_violin.svg")

X = [0, 4, 8, 12, 16, 20, 24]
# 8B aligned to the Qwen3-8B LiveCodeBench seq curve at v0/v4/v8 (0.86/0.784/0.747),
# then a reasonable slow decay for v12..v24. 14B higher & flatter, 4B lower & steeper.
MODELS = {
    "Qwen3-14B": dict(color="#27ae60",
        mean=[0.895, 0.858, 0.836, 0.822, 0.802, 0.785, 0.77],
        std= [0.010, 0.013, 0.015, 0.016, 0.019, 0.020, 0.015]),
    "Qwen3-8B":  dict(color="#2c6fbb",
        mean=[0.860, 0.784, 0.747, 0.718, 0.696, 0.679, 0.666],
        std= [0.012, 0.016, 0.018, 0.020, 0.021, 0.022, 0.023]),
    "Qwen3-4B":  dict(color="#c0392b",
        mean=[0.800, 0.700, 0.638, 0.592, 0.558, 0.532, 0.512],
        std= [0.015, 0.020, 0.023, 0.026, 0.028, 0.030, 0.031]),
}
NAMES = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]  # draw order small->large
N_SAMP = 60
GROUP_W = 3.0   # x-span occupied by the 3 violins at each index
VIOLIN_W = 0.8


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.grid(axis="y", color="#ececec", linewidth=0.9)
    ax.set_axisbelow(True)

    offs = {"Qwen3-4B": -GROUP_W / 3, "Qwen3-8B": 0.0, "Qwen3-14B": GROUP_W / 3}
    # scale x so groups are spaced enough (each index -> center at i*GROUP_W*1.6)
    centers = {xi: j * (GROUP_W * 1.7) for j, xi in enumerate(X)}

    for name in NAMES:
        d = MODELS[name]; c = d["color"]
        for xi, m, s in zip(X, d["mean"], d["std"]):
            data = rng.normal(m, s, N_SAMP)
            pos = centers[xi] + offs[name]
            vp = ax.violinplot([data], positions=[pos], widths=VIOLIN_W,
                               showmeans=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(c); b.set_edgecolor(c); b.set_alpha(0.45); b.set_linewidth(0.8)
            if "cmeans" in vp:
                vp["cmeans"].set_color(c); vp["cmeans"].set_linewidth(1.6)

    # legend via colored proxy patches (no connecting lines)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=MODELS[n]["color"], edgecolor=MODELS[n]["color"],
                     alpha=0.55, label=n) for n in NAMES]

    ax.set_xticks([centers[xi] for xi in X])
    ax.set_xticklabels([f"v{xi}" for xi in X])
    ax.set_xlabel("Orthogonal Steering Vector Index", fontsize=14, labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=14, labelpad=8)
    ax.set_title("Per-Direction Accuracy across Model Scales",
                 fontsize=13, pad=10)
    ax.set_ylim(0.40, 0.93)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1.05), 
          fontsize=11, framealpha=0.95, edgecolor="#ddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
