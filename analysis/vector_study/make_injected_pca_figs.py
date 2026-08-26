#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figures for the causal "does injecting the vector disturb the
hidden principal components?" analysis. Reads out/injected_pca_shift_<tag>.csv.

Two figures:
  (A) 3-panel summary vs layer:
        - subspace overlap (top-k PC subspace stability, 1 = unchanged)
        - variance-spectrum ratio  λ'/λ  (1 = unchanged)
        - hidden-cloud shift ‖Δmean‖/‖mean‖
      -> mid/low layers: overlap≈1, ratio≈1, shift≈0 (no disturbance);
         only the top layers (32-34) break this.
  (B) drift-in-principal-subspace vs layer (log y), against the random baseline
      -> the injected shift lives OUTSIDE the principal subspace at mid layers,
         but pours INTO it at the very top layers.

Rerun:  MODEL_TAG=qwen3-4b python make_injected_pca_figs.py
Output: figs/opd_injected-pca-summary.svg, figs/opd_injected-pca-drift.svg
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TAG = os.environ.get("MODEL_TAG", "qwen3-4b")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", f"injected_pca_shift_{_TAG}.csv")
FIG_A = os.path.join(HERE, "figs", "opd_injected-pca-summary.svg")
FIG_B = os.path.join(HERE, "figs", "opd_injected-pca-drift.svg")

C_MAIN = "#2ca02c"
C_HI = "#d62728"     # highlight high-layer regime
C_RAND = "#888888"
HI_START = 32        # layers >= this are the "token-manipulation" high regime


def _rc():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "figure.dpi": 400, "savefig.dpi": 400,
    })


def _load():
    rows = list(csv.DictReader(open(CSV)))
    rows.sort(key=lambda r: int(r["layer"]))
    d = {k: np.array([float(r[k]) for r in rows]) for k in
         ["layer", "subspace_overlap", "lambda_ratio_topk", "hidden_shift_rel",
          "drift_in_topk", "drift_rand_baseline"]}
    return d


def fig_a():
    d = _load()
    L = d["layer"]
    _rc()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))

    panels = [
        ("subspace_overlap", "Top-20 PC subspace overlap", "unchanged = 1", (None, 1.02)),
        ("lambda_ratio_topk", "Variance-spectrum ratio  λ'/λ", "unchanged = 1", None),
        ("hidden_shift_rel", "Hidden-cloud shift  ‖Δμ‖/‖μ‖", "no shift = 0", None),
    ]
    for ax, (key, ylab, ref, ylim) in zip(axes, panels):
        y = d[key]
        hi = L >= HI_START
        ax.plot(L, y, "-", color="#bbbbbb", lw=1.4, zorder=2)
        ax.scatter(L[~hi], y[~hi], s=52, color=C_MAIN, edgecolors="white", linewidths=1.1,
                   zorder=4, label="low/mid layers")
        ax.scatter(L[hi], y[hi], s=70, color=C_HI, edgecolors="white", linewidths=1.2,
                   marker="D", zorder=5, label="high layers (≥32)")
        if key in ("subspace_overlap", "lambda_ratio_topk"):
            ax.axhline(1.0, color=C_RAND, ls="--", lw=1.5, zorder=1)
        else:
            ax.axhline(0.0, color=C_RAND, ls="--", lw=1.5, zorder=1)
        ax.set_xlabel("Layer", fontsize=14, labelpad=6)
        ax.set_title(ylab, fontsize=13, pad=8)
        ax.text(0.03, 0.06, ref, transform=ax.transAxes, fontsize=10, color="#777", style="italic")
        if ylim:
            ax.set_ylim(*ylim)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="lower left", fontsize=9.5, framealpha=0.95, edgecolor="#dddddd")
    fig.suptitle("Injecting the Steering Vector Barely Perturbs the Hidden Principal Components",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_A), exist_ok=True)
    fig.savefig(FIG_A, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG_A}")
    plt.close(fig)


def fig_b():
    d = _load()
    L = d["layer"]; drift = d["drift_in_topk"]; rand = d["drift_rand_baseline"]
    _rc()
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    hi = L >= HI_START
    ax.axhline(float(rand[0]), color=C_RAND, ls="--", lw=1.6, zorder=1,
               label=f"random baseline ({rand[0]:.3f})")
    ax.plot(L, drift, "-", color="#bbbbbb", lw=1.4, zorder=2)
    ax.scatter(L[~hi], drift[~hi], s=60, color=C_MAIN, edgecolors="white", linewidths=1.2,
               zorder=4, label="low/mid layers")
    ax.scatter(L[hi], drift[hi], s=95, color=C_HI, edgecolors="white", linewidths=1.3,
               marker="D", zorder=5, label="high layers (≥32)")
    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=15, labelpad=8)
    ax.set_ylabel("Fraction of injected shift\ninside the principal subspace", fontsize=13.5, labelpad=8)
    ax.set_title("The Steered Direction Lives OUTSIDE the Principal Subspace", fontsize=13.5, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95, edgecolor="#dddddd")
    fig.tight_layout()
    fig.savefig(FIG_B, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG_B}")
    plt.close(fig)


def main():
    fig_a()
    fig_b()


if __name__ == "__main__":
    main()
