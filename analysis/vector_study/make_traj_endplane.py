#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project both v0 training trajectories onto the plane spanned by their two ENDPOINTS.

Instead of a distance-distorting t-SNE, we anchor on the answer: take the two converged v0
directions (off-policy end, on-policy end), which are cos=0.81 apart, and use them to build an
orthonormal 2-D basis. Every step of both (unit-normalized) trajectories is then linearly
projected onto that plane. Because the projection preserves inner products, the two endpoints
sit close together (they really are, cos 0.81) and the earlier trajectory points are shown
converging into that shared endpoint region — an honest, linear view.

Output: figs/opd_traj_endplane.svg
"""
import os, glob, re
import numpy as np
import torch

OFF = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors/singlev_layers5-15_lr5e-2"
ON = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/deepseek1p5b_rl_opd/trainable_vectors_seqbasis/repre_deepseek1p5b_rl_opd_seqbasis64_layers5-20_lr5e-2"
OUTFIG = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs/opd_traj_endplane.svg"
V0_MAX_STEP = 100
LY = list(range(5, 16))


def load_traj(run_dir):
    fs = sorted(glob.glob(f"{run_dir}/vectors_step_*.pt"),
                key=lambda p: int(re.search(r"(\d+)\.pt", p).group(1)))
    steps, V = [], []
    for f in fs:
        st = int(re.search(r"(\d+)\.pt", f).group(1))
        if st > V0_MAX_STEP:
            break
        o = torch.load(f, map_location="cpu", weights_only=False)
        if not all(L in o for L in LY):
            continue
        v = torch.cat([o[L].float().reshape(-1) for L in LY]).numpy()
        n = np.linalg.norm(v)
        if n > 1e-8:
            v = v / n
        steps.append(st); V.append(v)
    return np.array(steps), np.array(V)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_off, Voff = load_traj(OFF)
    s_on, Von = load_traj(ON)

    e_off = Voff[-1]; e_on = Von[-1]
    print(f"cos(end_off, end_on) = {float(e_off @ e_on):.4f}")

    # orthonormal 2-D basis from the two endpoints (Gram-Schmidt)
    u1 = e_off / np.linalg.norm(e_off)
    w = e_on - (e_on @ u1) * u1
    u2 = w / np.linalg.norm(w)
    B = np.vstack([u1, u2])                    # [2, D]

    off2 = Voff @ B.T                          # [T,2]
    on2 = Von @ B.T

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    ax.plot(off2[:, 0], off2[:, 1], "-", color="#2c6fbb", lw=1.8, alpha=0.5, zorder=2)
    ax.scatter(off2[:, 0], off2[:, 1], c=s_off, cmap="Blues", s=45, edgecolors="white",
               linewidths=0.5, zorder=4, label="off-policy distill")
    ax.plot(on2[:, 0], on2[:, 1], "-", color="#c0392b", lw=1.8, alpha=0.5, zorder=2)
    ax.scatter(on2[:, 0], on2[:, 1], c=s_on, cmap="Reds", s=45, edgecolors="white",
               linewidths=0.5, zorder=4, label="on-policy RL")

    # endpoints
    ax.scatter(*off2[-1], s=180, marker="*", color="#1a3c6e", edgecolors="white",
               linewidths=1.4, zorder=6)
    ax.scatter(*on2[-1], s=180, marker="*", color="#7b1f1f", edgecolors="white",
               linewidths=1.4, zorder=6)
    ax.annotate("converged\ndirections\n(cos=0.81)",
                xy=((off2[-1, 0] + on2[-1, 0]) / 2, (off2[-1, 1] + on2[-1, 1]) / 2),
                xytext=(0.62, 0.62), fontsize=10.5, color="#333", ha="center")

    ax.set_xlabel("endpoint basis  u1  (off-policy converged dir)", fontsize=11.5, labelpad=6)
    ax.set_ylabel("endpoint basis  u2", fontsize=11.5, labelpad=6)
    ax.set_title("Both Regimes Converge to the Same v0 Direction\n"
                 "(trajectories projected onto the endpoints' plane)", fontsize=11.5, pad=10)
    ax.axhline(0, color="#ddd", lw=0.8, zorder=0); ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.95, edgecolor="#ddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)
    fig.savefig(OUTFIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {OUTFIG}")


if __name__ == "__main__":
    main()
