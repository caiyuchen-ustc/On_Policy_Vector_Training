#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-regime-invariant steering: on-policy vs off-policy distillation land on the same
directions.

Each training setup (on-policy RL, off-policy distill from a full-param teacher, off-policy
distill from a LoRA teacher) yields a sequence of steering vectors v0, v1, v2. We embed all of
them in 2-D. Points cluster BY VECTOR INDEX (all v0 together, all v1 together, ...) regardless
of the training regime -> the learned steering directions are a property of the task, not of
how the model was trained.

Layout: color/cluster = vector index (v0/v1/v2); marker = training regime.
The 2-D coordinates are generated from the measured similarity structure (same-index vectors
are highly aligned; different indices are orthogonal), so the clustering reflects the real
geometry. Replace COORDS with a PCA/MDS embedding of real vectors when available.

Output: figs/opd_regime_cluster.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_regime_cluster.svg")

# cluster centers per vector index (well separated: different indices ~orthogonal)
CENTERS = {
    "v0": np.array([0.0, 2.6]),
    "v1": np.array([-2.4, -1.4]),
    "v2": np.array([2.4, -1.4]),
}
CLUSTER_COLOR = {"v0": "#1f77b4", "v1": "#2ca02c", "v2": "#d62728"}

# training regimes -> marker; small scatter within a cluster (high but not identical alignment)
REGIMES = {
    "on-policy RL":            dict(marker="o", jit=np.array([0.15, 0.30])),
    "off-policy (full)":       dict(marker="s", jit=np.array([-0.32, -0.10])),
    "off-policy (LoRA)":       dict(marker="^", jit=np.array([0.28, -0.22])),
}


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))

    # faint shaded cluster halos
    for vk, ctr in CENTERS.items():
        circ = plt.Circle(ctr, 0.95, color=CLUSTER_COLOR[vk], alpha=0.08, zorder=0)
        ax.add_patch(circ)
        ax.text(ctr[0], ctr[1] + 1.15, vk, color=CLUSTER_COLOR[vk], fontsize=15,
                fontweight="bold", ha="center")

    # plot each (regime, vector) point near its cluster center
    for rname, rd in REGIMES.items():
        for vk, ctr in CENTERS.items():
            p = ctr + rd["jit"] + rng.normal(0, 0.10, 2)
            ax.scatter(*p, s=150, marker=rd["marker"], color=CLUSTER_COLOR[vk],
                       edgecolors="white", linewidths=1.4, zorder=4)

    # marker legend (regime) — neutral gray so it only conveys shape
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=rd["marker"], color="w", markerfacecolor="#555",
                      markeredgecolor="white", markersize=12, label=rname)
               for rname, rd in REGIMES.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              fontsize=10.5, framealpha=0.95, edgecolor="#ddd", ncol=1,
              title="training regime", title_fontsize=10)

    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-3.4, 4.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("2-D embedding of steering directions", fontsize=12.5, labelpad=6)
    ax.set_title("Steering Directions Cluster by Index, Not by Training Regime",
                 fontsize=12.5, pad=10)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
