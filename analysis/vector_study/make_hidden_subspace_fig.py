#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-quality redraw of the hidden-subspace analysis (task 5).

Reads out/hidden_subspace.csv (already computed) and plots, per injected layer:
  (left)  E_top20 / random : vector energy in the hidden top-20 principal
          subspace, relative to a random vector. ~1 means "no more aligned with
          the high-variance directions than chance".
  (right) PC centroid rank : where the vector sits in the hidden PCA spectrum
          (low = principal / high-variance, high = tail / low-variance). The
          random-vector baseline is ~64.5.

Takeaway: across all layers the trained vector sits essentially where a RANDOM
vector would (centroid ≈ baseline, energy ≈ 1x) -> it does NOT ride the hidden
state's principal directions; it steers a low-variance, off-principal direction.

Rerun:  python make_hidden_subspace_fig.py
Output: fig/opd_hidden-subspace.svg
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", "hidden_subspace.csv")
FIG = os.path.join(HERE, "figs", "opd_hidden-subspace.svg")

C_ENERGY = "#d62728"   # red
C_RANK   = "#1f77b4"   # blue
C_RAND   = "#888888"


def _rc():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "figure.dpi": 400, "savefig.dpi": 400,
    })


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r.get("has_vec") == "1"]
    rows.sort(key=lambda r: int(r["layer"]))
    L = np.array([int(r["layer"]) for r in rows])
    E = np.array([float(r["E_top20_over_rand"]) for r in rows])
    rank = np.array([float(r["pc_centroid_rank"]) for r in rows])
    rand_rank = float(rows[0]["pc_centroid_rand"])

    _rc()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    # ---- left: energy in principal subspace ----
    ax = axes[0]
    ax.axhspan(0.0, 1.0, color="#eef3fb", alpha=0.9, zorder=0)          # "off-principal" band
    ax.axhline(1.0, color=C_RAND, ls="--", lw=1.8, zorder=2, label="random vector")
    ax.plot(L, E, "-", color=C_ENERGY, lw=1.6, alpha=0.5, zorder=3)
    ax.scatter(L, E, s=55, color=C_ENERGY, edgecolors="white", linewidths=1.2, zorder=4)
    ax.set_xlabel("Layer", fontsize=15, labelpad=8)
    ax.set_ylabel("Energy in hidden top-20 PC  (/ random)", fontsize=13.5, labelpad=8)
    ax.set_title("Vector energy in the hidden principal subspace", fontsize=13.5, pad=10)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    ax.margins(y=0.08)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="#dddddd")

    # ---- right: where in the spectrum does v live ----
    ax = axes[1]
    ax.axhline(rand_rank, color=C_RAND, ls="--", lw=1.8, zorder=2,
               label=f"random vector (≈{rand_rank:.0f})")
    ax.plot(L, rank, "-", color=C_RANK, lw=1.6, alpha=0.5, zorder=3)
    ax.scatter(L, rank, s=55, color=C_RANK, edgecolors="white", linewidths=1.2, zorder=4,
               label="trained vector")
    ax.set_xlabel("Layer", fontsize=15, labelpad=8)
    ax.set_ylabel("PC centroid rank  (low = principal)", fontsize=13.5, labelpad=8)
    ax.set_title("Where in the hidden spectrum does the vector live?", fontsize=13.5, pad=10)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    ax.invert_yaxis()   # principal (low rank) at top -> intuitive "high in spectrum"
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="#dddddd")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")
    print(f"E_top20/rand: mean={E.mean():.2f}  |  centroid rank: mean={rank.mean():.1f} (rand {rand_rank})")


if __name__ == "__main__":
    main()
