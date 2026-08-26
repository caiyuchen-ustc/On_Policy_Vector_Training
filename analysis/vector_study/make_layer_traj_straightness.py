#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-layer training-trajectory straightness (gapped-step direction cos) for v0/v1/v2.

For each trained vector and each layer, we sample its training window every GAP steps and
average cos between consecutive samples — a measure of how "straight" the direction moves
(high = direction barely turns; low = trajectory wanders). Averaged over the four SciKnowEval
domains. Findings (real data, layers 5-15, gap=20):
  - the LAST layers (14/15) are always the straightest, for every vector;
  - v0 shows a clear U-shape (middle layers wander most) and the lowest values overall
    (it learns the primary direction from scratch, so it turns the most);
  - v1/v2 are higher and flatter (trained in the frozen orthogonal complement -> smaller,
    more direct moves), the U-shape mostly washes out.
Output: figs/opd_layer_traj_straightness.svg
"""
import os, glob, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_layer_traj_straightness.svg")
ROOT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors"
DIRS = {"physics": "seqbasis_physics_layers5-15_lr5e-2",
        "chemistry": "seqbasis_chemistry_layers5-15_lr5e-2",
        "biology": "seqbasis_biology_layers5-15_lr5e-2",
        "material": "seqbasis_material_layers5-15_lr5e-2"}
LY = list(range(5, 16))
GAP = 20

VECS = [("v0", 0, "#2c6fbb"), ("v1", 1, "#e07b39"), ("v2", 2, "#4a9d5b")]


def load(base, s):
    o = torch.load(f"{base}/vectors_step_{s:07d}.pt", map_location="cpu", weights_only=False)
    return {L: o[L].float().numpy() for L in LY}


def windows(base):
    steps = sorted(int(re.findall(r"(\d+)", os.path.basename(f))[0])
                   for f in glob.glob(f"{base}/vectors_step_*.pt"))
    def flat(d): return np.concatenate([d[L] for L in LY])
    prev = flat(load(base, steps[0])); sw = []
    for s in steps[1:]:
        v = flat(load(base, s)); c = float(prev @ v / (np.linalg.norm(prev) * np.linalg.norm(v) + 1e-9))
        if c < 0.5: sw.append(s)
        prev = v
    bounds = [1] + sw + [steps[-1] + 1]
    wins = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
    return steps, wins


def gap_cos(base, steps, lo, hi, gap):
    ss = [s for s in steps if lo <= s <= hi][::gap]
    if len(ss) < 3:
        return None
    acc = {L: [] for L in LY}; prev = load(base, ss[0])
    for s in ss[1:]:
        d = load(base, s)
        for L in LY:
            a, b = prev[L], d[L]
            acc[L].append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        prev = d
    return np.array([np.mean(acc[L]) for L in LY])


def main():
    curves = {}
    for name, vidx, _ in VECS:
        rows = []
        for dom, sub in DIRS.items():
            base = f"{ROOT}/{sub}"; steps, wins = windows(base)
            if vidx >= len(wins):
                continue
            lo, hi = wins[vidx]; r = gap_cos(base, steps, lo, hi, GAP)
            if r is not None:
                rows.append(r)
        curves[name] = np.mean(rows, axis=0)

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.grid(axis="y", color="#ececec", lw=0.9); ax.set_axisbelow(True)

    x = np.array(LY)
    for name, vidx, col in VECS:
        if name not in curves:
            continue
        ax.plot(x, curves[name], "-o", color=col, lw=2.3, ms=6,
                mfc=col, mec="white", mew=1.2, zorder=4, label=name)

    ax.set_xlabel("Layer", fontsize=14, labelpad=8)
    ax.set_ylabel(f"Trajectory straightness\n(direction cos over {GAP}-step gap)", fontsize=13, labelpad=8)
    ax.set_title("Deep Layers Move Straightest; $v_0$ Wanders Most in the Middle",
                 fontsize=12.5, pad=10)
    ax.set_xticks(x)
    ax.set_ylim(0.72, 0.95)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95, edgecolor="#ddd",
              title="steering vector")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
