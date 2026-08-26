#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1: only ONE primary steering direction matters.

Sequential orthogonal basis: each vector v0, v1, v2, ... is trained one at a time (each
orthogonal to the previous ones), and evaluated SOLO right when it finishes (the eval injects
just that vector). Plotting solo performance vs vector index shows v0 dominates and later
orthogonal directions add little — there is effectively a single primary steering direction.

Data are the per-switch validation scores read from the seqbasis training logs
(ifevalg 7B, sciknoweval 1.5B).

Output: figs/opd_seq-solo-perf.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_seq-solo-perf.svg")

# solo val score of v0, v1, v2, ... read from training logs (eval fires on each vector switch)
SERIES = {
    "7B (DeepSeek-R1-Distill-Qwen-7B)": {
        "color": "#c0392b", "marker": "o",
        "perf": [0.4775, 0.4000, 0.3150, 0.3725, 0.3525],
        "baseline": 0.13,   # no-steer / base val (ifevalg)
    },
    "1.5B (DeepSeek-R1-Distill-Qwen-1.5B)": {
        "color": "#2c6fbb", "marker": "s",
        "perf": [0.5925, 0.5400, 0.5625, 0.5175, 0.4950, 0.5425],
        "baseline": 0.42,   # no-steer / base val (sciknoweval)
    },
}


def _rc():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 400, "savefig.dpi": 400,
    })


def main():
    _rc()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    for name, d in SERIES.items():
        y = d["perf"]
        x = np.arange(len(y))
        ax.plot(x, y, d["marker"] + "-", color=d["color"], lw=2.2, ms=9,
                markeredgecolor="white", markeredgewidth=1.2, label=name, zorder=4)
        # no-steer baseline as a faint dashed line in the same color
        ax.axhline(d["baseline"], color=d["color"], ls=":", lw=1.4, alpha=0.55, zorder=2)
        # annotate the v0 peak
        ax.annotate("only v0 helps", xy=(0, y[0]), xytext=(0.7, y[0] + 0.02),
                    fontsize=11, color=d["color"], ha="left")

    # shade "v0 dominates" region
    ax.axvspan(-0.3, 0.3, color="#fce9e6", alpha=0.7, zorder=0)

    ax.set_xlabel("orthogonal steering vector index", fontsize=15, labelpad=8)
    ax.set_ylabel("solo validation score", fontsize=15, labelpad=8)
    ax.set_title("Only One Primary Steering Direction Matters", fontsize=14, pad=10)
    xmax = max(len(d["perf"]) for d in SERIES.values())
    ax.set_xticks(range(xmax))
    ax.set_xticklabels([f"v{i}" for i in range(xmax)])
    ax.set_ylim(0.08, 0.66)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # legend + baseline note
    ax.plot([], [], ls=":", color="#888", label="no-steer baseline")
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.95, edgecolor="#dddddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
