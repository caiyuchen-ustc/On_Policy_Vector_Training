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

Canvas/fonts come from ../opd_plots/panel_style.py (via _panel_style). This is
the one 1x2 row in the group, so it uses save_wide(): each subplot gets exactly
the same axes size as a standalone panel, on a double-width canvas.

Rerun:  python make_hidden_subspace_fig.py
Output: figs/opd_hidden-subspace.svg
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _panel_style import (FS_AXLABEL, FS_LEGEND, FS_TITLE, WIDE_FIGSIZE,
                          panel_rc, save_wide)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", "hidden_subspace.csv")
FIG = os.path.join(HERE, "figs", "opd_hidden-subspace.svg")

C_ENERGY = "#d62728"   # red
C_RANK   = "#1f77b4"   # blue
C_RAND   = "#888888"


def _rc():
    # Canvas and fonts come from panel_style so this row matches the rest.
    panel_rc()


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r.get("has_vec") == "1"]
    rows.sort(key=lambda r: int(r["layer"]))
    L = np.array([int(r["layer"]) for r in rows])
    E = np.array([float(r["E_top20_over_rand"]) for r in rows])
    rank = np.array([float(r["pc_centroid_rank"]) for r in rows])
    rand_rank = float(rows[0]["pc_centroid_rand"])

    _rc()
    fig, axes = plt.subplots(1, 2, figsize=WIDE_FIGSIZE)

    # ---- left: energy in principal subspace ----
    ax = axes[0]
    ax.axhspan(0.0, 1.0, color="#eef3fb", alpha=0.9, zorder=0)          # "off-principal" band
    ax.axhline(1.0, color=C_RAND, ls="--", lw=1.4, zorder=2, label="Random vector")
    # Markers scale with the canvas: each subplot is a 229 pt panel, not half of
    # a 12.6 in figure, so s=55 would fuse the per-layer points together.
    ax.plot(L, E, "-", color=C_ENERGY, lw=1.0, alpha=0.5, zorder=3)
    ax.scatter(L, E, s=24, color=C_ENERGY, edgecolors="white", linewidths=0.7, zorder=4)
    ax.set_xlabel("Layer", fontsize=FS_AXLABEL, labelpad=6)
    # Both labels are held under the 148 pt axis height (the originals were 245
    # and 215 pt) and both titles under the 229 pt axes width (271 and 299 pt).
    ax.set_ylabel("Energy / random vector", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title("Energy in the Principal Subspace", fontsize=FS_TITLE, pad=9)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    ax.margins(y=0.08)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.0, handletextpad=0.35,
              labelspacing=0.22, borderpad=0.4)

    # ---- right: where in the spectrum does v live ----
    ax = axes[1]
    ax.axhline(rand_rank, color=C_RAND, ls="--", lw=1.4, zorder=2,
               label=f"Random vector (≈{rand_rank:.0f})")
    ax.plot(L, rank, "-", color=C_RANK, lw=1.0, alpha=0.5, zorder=3)
    ax.scatter(L, rank, s=24, color=C_RANK, edgecolors="white", linewidths=0.7, zorder=4,
               label="Trained vector")
    ax.set_xlabel("Layer", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel("PC centroid rank", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title("Position in the Hidden Spectrum", fontsize=FS_TITLE, pad=9)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    ax.invert_yaxis()   # principal (low rank) at top -> intuitive "high in spectrum"
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.0, handletextpad=0.35,
              labelspacing=0.22, borderpad=0.4)

    save_wide(fig, FIG)
    print(f"E_top20/rand: mean={E.mean():.2f}  |  centroid rank: mean={rank.mean():.1f} (rand {rand_rank})")


if __name__ == "__main__":
    main()
