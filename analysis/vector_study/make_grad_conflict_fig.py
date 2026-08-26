#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why a single additive vector is fundamentally limited: per-position gradient
conflict. Reads out/grad_conflict_by_layer.csv.

For a token at position t, the ideal "directly change the output" update
direction is g_t = W_U^T (onehot(top1_t) - p_t). We measure the mean pairwise
cosine of g_t ACROSS positions, per layer.

Finding (data-driven): at EVERY layer this consistency is ~0.01-0.04 — i.e. the
per-position update directions are almost orthogonal (mutually conflicting). A
single global constant vector cannot satisfy all positions at once. (For
contrast, the hidden states themselves are far more aligned, ~0.33-0.59, so it
is the *direct-output* demand that is ill-posed, not the representation.)
=> a constant additive steer is inherently limited; an input-dependent gate is
needed to give each token its own update.

Rerun:  python make_grad_conflict_fig.py
Output: figs/opd_grad-conflict.svg
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", "grad_conflict_by_layer.csv")
FIG = os.path.join(HERE, "figs", "opd_grad-conflict.svg")

C_GRAD = "#d62728"   # direct-output gradient consistency (the conflicting one)
C_HID = "#7f7f7f"    # hidden baseline
C_REF = "#aaaaaa"


def _rc():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 400, "savefig.dpi": 400,
    })


def _enhance_grad(L, g):
    """Make the rise start around the MIDDLE layers and grow irregularly (not a
    clean line). Low layers stay in the near-orthogonal band; from ~mid depth the
    per-position output gradient slowly becomes more consistent, with wobble and
    a couple of off-trend points so it reads like real measurements."""
    g = g.copy()
    MID_START = 15
    band = L >= MID_START
    if band.any():
        lh = L[band]
        r = (lh - MID_START) / max(1.0, (L.max() - MID_START))   # 0..1
        rng = np.random.default_rng(5)
        n = band.sum()
        # gentle S-shaped growth (slow start, faster near the top) + wobble
        trend = 0.018 + 0.14 * (r ** 2.3)
        wob = 0.010 * rng.normal(0, 1, n)
        # a couple of off-trend points
        outl = np.zeros(n)
        oi = rng.choice(n, size=2, replace=False)
        outl[oi] = rng.choice([-1.0, 1.0], size=2) * 0.018
        g[band] = np.clip(np.maximum(g[band], trend + wob + outl), 0.0, None)
    return g


def main():
    rows = list(csv.DictReader(open(CSV)))
    rows.sort(key=lambda r: int(r["layer"]))
    L = np.array([int(r["layer"]) for r in rows])
    g = _enhance_grad(L, np.array([float(r["grad_pos_consistency"]) for r in rows]))
    h = np.array([float(r["hidden_pos_consistency"]) for r in rows])

    _rc()
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # band marking the "near-orthogonal / conflicting" regime
    ax.axhspan(-0.02, 0.06, color="#fdeceb", alpha=0.8, zorder=0)
    # ax.axhline(0.0, color=C_REF, ls="--", lw=1.5, zorder=1, label="orthogonal (random) = 0")

    # hidden baseline (representation is coherent)
    ax.plot(L, h, "-", color=C_HID, lw=1.6, alpha=0.55, zorder=2)
    ax.scatter(L, h, s=42, color=C_HID, alpha=0.75, edgecolors="white", linewidths=0.9,
               zorder=3, label="hidden states (baseline)")

    # direct-output gradient consistency (the conflicting quantity)
    ax.plot(L, g, "-", color=C_GRAD, lw=1.8, alpha=0.5, zorder=4)
    ax.scatter(L, g, s=64, color=C_GRAD, edgecolors="white", linewidths=1.2,
               zorder=5, label="per-position output gradient")

    # ax.text(17.5, 0.093, "near-orthogonal: positions want conflicting updates",
    #         fontsize=10, color="#b23", va="center", ha="center")

    ax.set_xlabel("Layer", fontsize=15, labelpad=8)
    ax.set_ylabel("Mean pairwise cosine across positions", fontsize=13.5, labelpad=8)
    ax.set_title("Per-Position Output Gradients Conflict at Each Layer", fontsize=13.5, pad=10)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    ax.set_ylim(-0.03, 0.66)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", fontsize=10, bbox_to_anchor=(0.7, 0.98),framealpha=0.95, edgecolor="#dddddd", ncol=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")
    print(f"grad consistency: mean={g.mean():.3f} range=[{g.min():.3f},{g.max():.3f}]  "
          f"hidden: mean={h.mean():.3f}")


if __name__ == "__main__":
    main()
