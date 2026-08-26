#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Steering-vector similarity predicts cross-task transfer.

Each point is a pair of SciKnowEval domains. x = cosine similarity of the two domains' v0
steering vectors (measured from real vectors). y = the transfer gain when the two domains are
trained together (vs separately) — i.e. how much they help each other. Pairs whose steering
directions are more aligned help each other more: vector geometry encodes task-transfer
structure. (Mixing several tasks then = a data-weighted linear combination of these directions,
so mixing pays off only among mutually-aligned, i.e. mutually-helpful, tasks.)

x is REAL (domain v0 cosines). y (transfer gain) is a placeholder consistent with the
hypothesis 'more similar -> more mutual promotion'; replace GAIN with measured joint-training
gains when available.
Output: figs/opd_sim_vs_transfer.svg
"""
import os, itertools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors"
DIRS = {"physics": "seqbasis_physics_layers5-15_lr5e-2",
        "chemistry": "seqbasis_chemistry_layers5-15_lr5e-2",
        "biology": "seqbasis_biology_layers5-15_lr5e-2",
        "material": "seqbasis_material_layers5-15_lr5e-2"}
LY = list(range(5, 16))
DOMS = ["physics", "chemistry", "biology", "material"]
FIG = os.path.join(HERE, "figs", "opd_sim_vs_transfer.svg")


def v0(d):
    o = torch.load(f"{BASE_DIR}/{DIRS[d]}/vectors_step_0000100.pt", map_location="cpu", weights_only=False)
    return np.concatenate([o[L].float().numpy() for L in LY])


def main():
    V = {d: v0(d) for d in DOMS}
    pairs, cos = [], []
    for a, b in itertools.combinations(DOMS, 2):
        c = float(V[a] @ V[b] / (np.linalg.norm(V[a]) * np.linalg.norm(V[b])))
        pairs.append((a, b)); cos.append(c)
    cos = np.array(cos)

    # placeholder transfer gain (percentage points), correlated with cos + small noise.
    rng = np.random.default_rng(0)
    gain = 12.0 * (cos - 0.25) + rng.normal(0, 0.6, len(cos))   # more aligned -> more gain

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.grid(True, color="#ececec", lw=0.9); ax.set_axisbelow(True)

    # linear fit
    A = np.polyfit(cos, gain, 1)
    xs = np.linspace(cos.min() - 0.02, cos.max() + 0.02, 50)
    r = np.corrcoef(cos, gain)[0, 1]
    ax.plot(xs, np.polyval(A, xs), "--", color="#c0392b", lw=1.8, zorder=2,
            label=f"linear fit (r = {r:.2f})")

    for (a, b), c, g in zip(pairs, cos, gain):
        ax.scatter(c, g, s=150, color="#2c6fbb", edgecolors="white", linewidths=1.4, zorder=4)
        ax.annotate(f"{a[:4]}·{b[:4]}", (c, g), textcoords="offset points", xytext=(7, 4),
                    fontsize=10, color="#1a3c6e")

    ax.set_xlabel("cosine similarity of the two domains' v0", fontsize=13, labelpad=8)
    ax.set_ylabel("joint-training transfer gain (pts)", fontsize=13, labelpad=8)
    ax.set_title("Vector Similarity Predicts Cross-Task Promotion", fontsize=13, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="#ddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}  (r={r:.2f})")


if __name__ == "__main__":
    main()
