#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-layer training trajectory of the FIRST steering vector v0 (physics, steps 1..100).

For each layer L in 5..15, we track cos(v0^(t)[L], v0^(final)[L]) — how the direction at each
training step aligns with the finally-converged v0 direction of that layer. All layers converge
monotonically from ~0.15 to 1.0, but DEEPER layers converge FASTER (lead shallow layers by
~0.1-0.15 cos throughout), i.e. deep layers lock in the direction first and shallow layers follow.

Real data (layers 5-15, physics seqbasis run). Output: figs/opd_v0_layer_traj.svg
"""
import os, glob, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_v0_layer_traj.svg")
BASE = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors/seqbasis_physics_layers5-15_lr5e-2"
LY = list(range(5, 16))
V0_END = 100     # v0 training window is steps 1..100


def load(step):
    o = torch.load(f"{BASE}/vectors_step_{step:07d}.pt", map_location="cpu", weights_only=False)
    return {L: o[L].float().numpy() for L in LY}


def main():
    steps = sorted(int(re.findall(r"(\d+)", os.path.basename(f))[0])
                   for f in glob.glob(f"{BASE}/vectors_step_*.pt"))
    steps = [s for s in steps if s <= V0_END]
    final = load(V0_END)

    # cos(direction at step t, final direction) per layer
    cos = {L: [] for L in LY}
    for s in steps:
        d = load(s)
        for L in LY:
            fL = final[L]; dL = d[L]
            cos[L].append(float(dL @ fL / (np.linalg.norm(dL) * np.linalg.norm(fL) + 1e-9)))

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    # shallow -> deep : light -> dark blue
    cmap = LinearSegmentedColormap.from_list("bl", ["#b5d4ea", "#2c6fbb", "#123a63"])
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.grid(axis="y", color="#ececec", lw=0.9); ax.set_axisbelow(True)

    x = np.array(steps)
    for i, L in enumerate(LY):
        c = cmap(i / (len(LY) - 1))
        ax.plot(x, cos[L], "-", color=c, lw=2.0, zorder=3 + i, alpha=0.95)

    ax.set_xlabel("Training step (v0 window)", fontsize=14, labelpad=8)
    ax.set_ylabel(r"cos$(v_0^{(t)},\ v_0^{\mathrm{final}})$", fontsize=14, labelpad=8)
    ax.set_title("Deeper Layers Converge First: Per-Layer Trajectory of $v_0$",
                 fontsize=13, pad=10)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.02)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # colorbar as layer-depth legend
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=LY[0], vmax=LY[-1]))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, ticks=[LY[0], 10, LY[-1]])
    cb.set_label("Layer (shallow → deep)", fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
