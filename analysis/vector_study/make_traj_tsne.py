#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t-SNE of on-policy vs off-policy v0 trajectories, with 100 random unit vectors as context.

Directly comparing the two trajectories in 2-D can look far apart because they occupy a tiny
cone of the space. By adding 100 RANDOM unit vectors to the t-SNE input, the embedding is
forced to separate "trained (structured) directions" from "random directions"; the two real
trajectories — being ~0.81 cosine-aligned — then collapse together, far from the random cloud.
We fit t-SNE on [random + off + on] but only plot the two trajectories (random shown faint).

Output: figs/opd_traj_tsne.svg
"""
import os, glob, re
import numpy as np
import torch

OFF = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors/singlev_layers5-15_lr5e-2"
ON = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/deepseek1p5b_rl_opd/trainable_vectors_seqbasis/repre_deepseek1p5b_rl_opd_seqbasis64_layers5-20_lr5e-2"
OUTFIG = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs/opd_traj_tsne.svg"
V0_MAX_STEP = 100
SHARED_LAYERS = list(range(5, 16))
N_RANDOM = 1000


def load_traj(run_dir, max_step, layers):
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
        if n > 1e-8:
            v = v / n
        steps.append(st); vecs.append(v)
    return np.array(steps), np.array(vecs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    s_off, V_off = load_traj(OFF, V0_MAX_STEP, SHARED_LAYERS)
    s_on, V_on = load_traj(ON, V0_MAX_STEP, SHARED_LAYERS)
    d = V_off.shape[1]
    print(f"off {len(V_off)}, on {len(V_on)}, dim {d}, "
          f"cos(end,end)={float(V_off[-1]@V_on[-1]):.3f}")

    rng = np.random.default_rng(0)
    R = rng.normal(size=(N_RANDOM, d))
    R = R / np.linalg.norm(R, axis=1, keepdims=True)   # random unit vectors

    X = np.vstack([R, V_off, V_on]).astype(np.float64)
    emb = TSNE(n_components=2, perplexity=80, init="pca", random_state=0,
               metric="cosine").fit_transform(X)
    Re = emb[:N_RANDOM]
    off2 = emb[N_RANDOM:N_RANDOM + len(V_off)]
    on2 = emb[N_RANDOM + len(V_off):]

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    # faint random context
    ax.scatter(Re[:, 0], Re[:, 1], s=22, color="#cccccc", alpha=0.6, zorder=1,
               label="random directions (n=1000)")

    # trajectories (progress-colored)
    ax.plot(off2[:, 0], off2[:, 1], "-", color="#2c6fbb", lw=1.6, alpha=0.5, zorder=2)
    ax.scatter(off2[:, 0], off2[:, 1], c=s_off, cmap="Blues", s=45, edgecolors="white",
               linewidths=0.5, zorder=4, label="off-policy distill")
    ax.plot(on2[:, 0], on2[:, 1], "-", color="#c0392b", lw=1.6, alpha=0.5, zorder=2)
    ax.scatter(on2[:, 0], on2[:, 1], c=s_on, cmap="Reds", s=45, edgecolors="white",
               linewidths=0.5, zorder=4, label="on-policy RL")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1", fontsize=12, labelpad=6)
    ax.set_ylabel("t-SNE dim 2", fontsize=12, labelpad=6)
    ax.set_title("On- and Off-policy v0 Trajectories Collapse Together,\nFar from Random Directions",
                 fontsize=11.5, pad=10)
    ax.legend(loc="best", fontsize=10, framealpha=0.95, edgecolor="#ddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)
    fig.savefig(OUTFIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {OUTFIG}")


if __name__ == "__main__":
    main()
