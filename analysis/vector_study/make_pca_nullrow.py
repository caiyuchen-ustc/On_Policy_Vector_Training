#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix 1x3 row: the three injected-PCA null results on one canvas.

    subspace overlap ~= 1     the principal subspace is unchanged
    variance ratio   ~= 1     the spectrum is unchanged
    hidden shift     ~= 0     the activation cloud barely moves

These say "injecting the vector does NOT disturb the principal structure at
low/mid layers", i.e. three confirmatory non-effects that support the same
sentence. Individually they each cost a figure slot for one flat line; as a row
they cost one slot and the shared x axis makes the common break at the top
layers obvious. Only opd_pca-drift.svg has a positive trend, so it stays a
standalone panel and is not included here.

Data and per-panel styling are reused verbatim from make_injected_pca_figs2.py
(same _load/_enhance/_scatter_by_regime), so this row can never disagree with
the standalone panels.

Rerun:  python make_pca_nullrow.py
Output: figs/opd_pca-nullrow.svg
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _panel_style import FS_AXLABEL, FS_LEGEND, FS_TITLE, panel_rc, row_figsize, save_row
from make_injected_pca_figs2 import (FIGDIR, C_REF, _enhance, _load,
                                     _scatter_by_regime)

FIG = os.path.join(FIGDIR, "opd_pca-nullrow.svg")

# (key, kind, ylabel, title, ref_line, ylim) -- identical to the standalone panels.
PANELS = [
    ("subspace_overlap", "overlap", "Subspace overlap", "Principal Subspace Stability",
     1.0, (0.86, 1.005)),
    ("lambda_ratio_topk", "lambda", "Spectrum ratio  λ'/λ", "Variance Spectrum Change",
     1.0, (0.99, 1.02)),
    ("hidden_shift_rel", "shift", "Shift  ‖Δμ‖ / ‖μ‖", "Hidden-State Displacement",
     0.0, None),
]


def main():
    d = _load()
    L = d["layer"]
    panel_rc()
    fig, axes = plt.subplots(1, len(PANELS), figsize=row_figsize(len(PANELS)))

    for ax, (key, kind, ylabel, title, ref, ylim) in zip(axes, PANELS):
        y = _enhance(L, d[key], kind)
        if ref is not None:
            ax.axhline(ref, color=C_REF, ls="--", lw=1.4, zorder=1)
        _scatter_by_regime(ax, L, y)
        ax.set_xlabel("Layer", fontsize=FS_AXLABEL, labelpad=6)
        ax.set_ylabel(ylabel, fontsize=FS_AXLABEL, labelpad=6)
        ax.set_title(title, fontsize=FS_TITLE, pad=9)
        ax.set_xlim(L.min() - 1, L.max() + 1)
        if ylim:
            ax.set_ylim(*ylim)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # One legend for the row, not three: the regime colours are shared, so
    # repeating the key three times spends axes space on nothing. It goes on the
    # middle panel, whose lower left is empty in all three y ranges.
    axes[1].legend(loc="lower left", fontsize=FS_LEGEND, framealpha=0.95,
                   edgecolor="#dddddd", handlelength=1.0, handletextpad=0.35,
                   labelspacing=0.22, borderpad=0.4)
    save_row(fig, FIG)
    plt.close(fig)


if __name__ == "__main__":
    main()
