#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mixed-vs-per-domain steering analysis (1.5B, SciKnowEval, 4 domains). Real vectors.

Figures:
  A  data_weight   : per-domain training-data fraction  vs  that domain's weight in the mixed v0
                     (least-squares decomposition). Points hug y=x -> mixed training fuses the
                     domain directions in proportion to how much data each contributes.
  B  domain_sim    : 4x4 cosine similarity of the four domains' v0 (shared but distinct).
  C  mix_decomp    : the mixed v0 decomposed onto the four domain v0 (bar of |weights|, R^2).

All values computed from the real trained vectors (step-100 v0, layers 5-15).
Output: figs/opd_mix_{dataweight,domainsim,decomp}.svg
"""
import os, itertools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors"
DIRS = {"mixed": "singlev_layers5-15_lr5e-2",
        "physics": "seqbasis_physics_layers5-15_lr5e-2",
        "chemistry": "seqbasis_chemistry_layers5-15_lr5e-2",
        "biology": "seqbasis_biology_layers5-15_lr5e-2",
        "material": "seqbasis_material_layers5-15_lr5e-2"}
LY = list(range(5, 16))
DOMS = ["physics", "chemistry", "biology", "material"]
DCOLOR = {"physics": "#1f77b4", "chemistry": "#ff7f0e", "biology": "#2ca02c", "material": "#9467bd"}
TEACHER_N = {"physics": 14400, "chemistry": 37800, "biology": 9000, "material": 16820}


def v0(d):
    o = torch.load(f"{BASE_DIR}/{DIRS[d]}/vectors_step_0000100.pt", map_location="cpu", weights_only=False)
    return np.concatenate([o[L].float().numpy() for L in LY])


def _rc():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })


def main():
    V = {d: v0(d) for d in DOMS}
    M = v0("mixed")
    A = np.stack([V[d] for d in DOMS], 1)
    sol, *_ = np.linalg.lstsq(A, M, rcond=None)
    recon = A @ sol
    r2 = 1 - ((M - recon) ** 2).sum() / ((M - M.mean()) ** 2).sum()
    w = np.abs(sol) / np.abs(sol).sum()
    tot = sum(TEACHER_N.values())
    frac = np.array([TEACHER_N[d] / tot for d in DOMS])

    _rc()
    # ---------- A: data fraction vs mixed-weight ----------
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.grid(True, color="#ececec", lw=0.9); ax.set_axisbelow(True)
    lim = 0.55
    ax.plot([0, lim], [0, lim], "--", color="#999", lw=1.4, zorder=1, label="y = x")
    for i, d in enumerate(DOMS):
        ax.scatter(frac[i], w[i], s=170, color=DCOLOR[d], edgecolors="white", linewidths=1.5,
                   zorder=4, label=d)
        ax.annotate(d, (frac[i], w[i]), textcoords="offset points", xytext=(8, 4),
                    fontsize=10.5, color=DCOLOR[d])
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("training-data fraction of domain", fontsize=13, labelpad=8)
    ax.set_ylabel("weight in mixed v0 (decomposition)", fontsize=13, labelpad=8)
    ax.set_title("Mixed Training Fuses Domain Directions\nin Proportion to Data", fontsize=12.5, pad=10)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95, edgecolor="#ddd")
    fig.tight_layout(); fig.savefig(f"{HERE}/figs/opd_mix_dataweight.svg", bbox_inches="tight", format="svg", dpi=400)
    plt.close(fig)

    # ---------- B: 4x4 domain-domain cosine ----------
    S = np.eye(4)
    for i, j in itertools.combinations(range(4), 2):
        c = V[DOMS[i]] @ V[DOMS[j]] / (np.linalg.norm(V[DOMS[i]]) * np.linalg.norm(V[DOMS[j]]))
        S[i, j] = S[j, i] = c
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(S, cmap="YlGnBu", vmin=0.2, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(DOMS, rotation=20, ha="right"); ax.set_yticklabels(DOMS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=12,
                    color="white" if S[i, j] > 0.7 else "#222", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("cosine of v0", fontsize=11)
    ax.set_title("Domain Steering Directions:\nShared Yet Distinct", fontsize=12.5, pad=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/figs/opd_mix_domainsim.svg", bbox_inches="tight", format="svg", dpi=400)
    plt.close(fig)

    # ---------- C: mixed v0 decomposition bar ----------
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.grid(axis="y", color="#ececec", lw=0.9); ax.set_axisbelow(True)
    xs = np.arange(4)
    ax.bar(xs, w, width=0.6, color=[DCOLOR[d] for d in DOMS], edgecolor="white", linewidth=1.0)
    for i, wi in enumerate(w):
        ax.text(i, wi + 0.01, f"{wi:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(DOMS, rotation=20, ha="right")
    ax.set_ylabel("weight in mixed v0", fontsize=13, labelpad=8)
    ax.set_ylim(0, max(w) * 1.25)
    ax.set_title(f"Mixed v0 Decomposes onto Domain Directions  (R²={r2:.2f})", fontsize=12, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{HERE}/figs/opd_mix_decomp.svg", bbox_inches="tight", format="svg", dpi=400)
    plt.close(fig)

    print(f"[ok] saved 3 figs. R2={r2:.3f}, weights={dict(zip(DOMS, np.round(w,3)))}")


if __name__ == "__main__":
    main()
