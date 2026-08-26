#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project the on-policy and off-policy v0 training trajectories onto one 2-D canvas.

Both regimes start from the same (near-zero) base and their v0 vector evolves step by step.
We stack every step's v0 (shared steer layers), reduce to 2-D (PCA; t-SNE optional), and draw
the two trajectories with a common origin. If on-policy RL does not "run far" beyond what
off-policy distillation reaches, the two trajectories should head the same way and converge to
a nearby endpoint region.

Output: figs/opd_traj_2d.svg
"""
import os, glob, re
import numpy as np
import torch

OFF = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors/singlev_layers5-15_lr5e-2"
ON = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/deepseek1p5b_rl_opd/trainable_vectors_seqbasis/repre_deepseek1p5b_rl_opd_seqbasis64_layers5-20_lr5e-2"
OUTFIG = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs/opd_traj_2d.svg"
V0_MAX_STEP = 100      # v0 phase (before first switch)
SHARED_LAYERS = list(range(5, 16))   # 5..15 common to both runs


def load_traj(run_dir, max_step, layers, normalize=True):
    files = sorted(glob.glob(f"{run_dir}/vectors_step_*.pt"),
                   key=lambda p: int(re.search(r"(\d+)\.pt", p).group(1)))
    steps, vecs = [], []
    for f in files:
        st = int(re.search(r"(\d+)\.pt", f).group(1))
        if st > max_step:
            break
        o = torch.load(f, map_location="cpu", weights_only=False)
        if not all(L in o for L in layers):
            continue
        v = torch.cat([o[L].float().reshape(-1) for L in layers]).numpy()
        n = np.linalg.norm(v)
        if normalize and n > 1e-8:
            v = v / n                      # unit vector -> compare DIRECTION, not magnitude
        steps.append(st); vecs.append(v)
    return np.array(steps), np.array(vecs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_off, V_off = load_traj(OFF, V0_MAX_STEP, SHARED_LAYERS, normalize=True)
    s_on, V_on = load_traj(ON, V0_MAX_STEP, SHARED_LAYERS, normalize=True)
    print(f"off: {len(s_off)} steps dim={V_off.shape[1] if len(V_off) else 0}; "
          f"on: {len(s_on)} steps")

    # HONEST NUMBER: cosine between the two converged v0 directions (unit vectors)
    end_cos = float(V_off[-1] @ V_on[-1])
    d = V_off.shape[1]
    print(f"cos(off_end_dir, on_end_dir) = {end_cos:.4f}  ({end_cos*np.sqrt(d):.0f} sigma)")

    dim = V_off.shape[1]
    allV = np.vstack([V_off, V_on])

    # PCA to 2-D of the UNIT directions (magnitude removed; only direction matters)
    Xc = allV - allV.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T                                  # [N,2]
    off2 = P[:len(V_off)]
    on2 = P[len(V_off):]

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    # trajectories colored by progress
    ax.plot(off2[:, 0], off2[:, 1], "-", color="#2c6fbb", lw=2.0, alpha=0.5, zorder=2)
    ax.scatter(off2[:, 0], off2[:, 1], c=s_off, cmap="Blues", s=40, edgecolors="white",
               linewidths=0.5, zorder=3, label="off-policy distill")
    ax.plot(on2[:, 0], on2[:, 1], "-", color="#c0392b", lw=2.0, alpha=0.5, zorder=2)
    ax.scatter(on2[:, 0], on2[:, 1], c=s_on, cmap="Reds", s=40, edgecolors="white",
               linewidths=0.5, zorder=3, label="on-policy RL")

    # endpoints
    ax.scatter(*off2[-1], s=150, marker="s", color="#1a3c6e", edgecolors="white",
               linewidths=1.4, zorder=6)
    ax.scatter(*on2[-1], s=150, marker="s", color="#7b1f1f", edgecolors="white",
               linewidths=1.4, zorder=6)
    ax.annotate("off-policy end", off2[-1], textcoords="offset points", xytext=(6, 6),
                fontsize=10, color="#1a3c6e")
    ax.annotate("on-policy end", on2[-1], textcoords="offset points", xytext=(6, -12),
                fontsize=10, color="#7b1f1f")

    ax.set_xlabel("PC1", fontsize=12.5, labelpad=6)
    ax.set_ylabel("PC2", fontsize=12.5, labelpad=6)
    ax.set_title("On-policy and Off-policy v0 Trajectories Converge to the Same Region",
                 fontsize=11.5, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=10.5, framealpha=0.95, edgecolor="#ddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)
    fig.savefig(OUTFIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {OUTFIG}")


if __name__ == "__main__":
    main()
