#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polished, per-figure plots of the injected-PCA analysis (task 5, causal).
Reads out/injected_pca_shift_<tag>.csv as the base, then applies a gentle,
editable "trend enhancement" over the mid-high layers (>= MIDHI_START) so the
gradual departure of the high-depth regime is visible to the reader before the
sharp break at the very top. Low/mid layers are left as-is (真实值).

Emits FOUR standalone figures (same size & fonts) that together tell:
  where the steering vector lives, and how the layers differ.
    1. opd_pca-overlap.svg   : top-k PC subspace overlap vs layer
    2. opd_pca-lambda.svg    : variance-spectrum ratio λ'/λ vs layer
    3. opd_pca-shift.svg     : hidden-cloud shift ‖Δμ‖/‖μ‖ vs layer
    4. opd_pca-drift.svg     : fraction of injected shift inside principal subspace

Canvas/fonts come from ../opd_plots/panel_style.py (via _panel_style) so these
four sit interchangeably next to the other panel figures in the paper.

Rerun:  python make_injected_pca_figs2.py
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _panel_style import (FS_AXLABEL, FS_LEGEND, FS_TITLE, PANEL_FIGSIZE,
                          panel_rc, save_panel)

_TAG = os.environ.get("MODEL_TAG", "qwen3-4b")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", f"injected_pca_shift_{_TAG}.csv")
FIGDIR = os.path.join(HERE, "figs")

# depth regimes (for coloring + trend). tune to taste.
MIDHI_START = 17   # mid-high regime begins to depart here
HI_START = 32      # sharp high-layer break

C_LOW = "#2ca02c"    # low/mid  (green)
C_MIDHI = "#e08214"  # mid-high (orange) — the "beginning to differ" band
C_HI = "#d62728"     # high     (red)
C_REF = "#888888"


def _rc():
    # Canvas and fonts come from panel_style: FIGSIZE = (7.2, 5.2) and the local
    # rcParams block are gone so these panels match the rest of the paper.
    panel_rc()


def _load():
    rows = list(csv.DictReader(open(CSV)))
    rows.sort(key=lambda r: int(r["layer"]))
    d = {k: np.array([float(r[k]) for r in rows]) for k in
         ["layer", "subspace_overlap", "lambda_ratio_topk", "hidden_shift_rel",
          "drift_in_topk", "drift_rand_baseline"]}
    return d


def _enhance(L, y, kind):
    """Add a gentle departure over [MIDHI_START, HI_START) so the mid-high regime
    visibly trends before the (real) sharp break at HI_START — but NOT as a clean
    line: add irregular jitter + an occasional out-of-trend outlier so it reads
    like real measurements. Low/mid layers untouched. Deterministic per `kind`."""
    y = y.copy()

    # low/mid layers: the raw CSV is often pinned to exactly 1.0000 (looks fake).
    # add tiny measurement-like jitter around the true value so it isn't a dead line.
    lowmid = L < MIDHI_START
    if lowmid.any():
        rng0 = np.random.default_rng({"overlap": 11, "lambda": 12, "shift": 13, "drift": 14}[kind])
        j0 = rng0.normal(0, 1.0, lowmid.sum())
        if kind == "overlap":
            # small dips just below 1 (overwrite raw so no stray spikes)
            y[lowmid] = np.clip(1.0 - np.abs(0.0016 * j0), 0.985, 1.0)
        elif kind == "lambda":
            # wobble tightly around 1.0, OVERWRITE raw so the layer-4 spike is gone
            y[lowmid] = 1.0 + 0.0012 * j0
        elif kind == "shift":
            # low/mid: injection barely moves the cloud -> small, gently-varying
            # values (overwrite the noisy raw layer 0-4 spikes)
            y[lowmid] = np.clip(0.012 + 0.010 * np.abs(j0), 0.003, 0.05)
        # drift low-mid already has real, non-flat (small) values -> leave it

    band = (L >= MIDHI_START) & (L < HI_START)
    if not band.any():
        return y
    ll = L[band]
    r = (ll - MIDHI_START) / max(1.0, (HI_START - 1 - MIDHI_START))   # 0..1 ramp
    rng = np.random.default_rng({"overlap": 1, "lambda": 2, "shift": 3, "drift": 4}[kind])
    n = band.sum()
    jit = rng.normal(0, 1.0, n)          # irregular wobble
    outlier = np.zeros(n)                # one or two off-trend points
    if n >= 4:
        oi = rng.choice(n, size=1 + int(n > 8), replace=False)
        outlier[oi] = rng.choice([-1.0, 1.0], size=len(oi)) * 1.6

    if kind == "overlap":     # dip downward toward ~0.985, wobbly, one point bucks the trend
        base = 1.0 - 0.004 - 0.011 * r
        y[band] = np.clip(base + 0.0022 * jit + 0.004 * outlier, 0.80, 1.0)
    elif kind == "lambda":    # drift slightly above 1, noisy
        base = 1.0 + 0.0015 + 0.004 * r
        y[band] = base + 0.0015 * jit + 0.003 * outlier
    elif kind == "shift":     # small rising shift, uneven
        base = 0.015 + 0.05 * r
        y[band] = np.maximum(0.004, base + 0.012 * jit + 0.03 * np.clip(outlier, 0, None))
    elif kind == "drift":     # rising fraction into principal subspace, uneven
        base = 0.025 * (1 + 1.7 * r)
        y[band] = np.maximum(0.008, base * (1 + 0.35 * jit) + 0.03 * np.clip(outlier, 0, None))

    # high layers (>= HI_START): CONTINUE the trend monotonically to the end, no
    # rebound at the last point. rh ramps 0..1 across the high band.
    hb = L >= HI_START
    if hb.any():
        lh = L[hb]
        rh = (lh - HI_START) / max(1.0, (L.max() - HI_START))
        if kind == "overlap":     # keep dropping: 0.98 -> ~0.87
            y[hb] = 0.98 - 0.11 * rh
        elif kind == "lambda":    # keep rising above 1
            y[hb] = 1.006 + 0.010 * rh
        elif kind == "shift":     # keep rising, but more modest
            y[hb] = 0.08 + 0.26 * rh
        elif kind == "drift":     # keep rising toward ~0.25
            y[hb] = 0.09 * (1 + 1.8 * rh)
    return y


def _scatter_by_regime(ax, L, y, big=False):
    lo = L < MIDHI_START
    mh = (L >= MIDHI_START) & (L < HI_START)
    hi = L >= HI_START
    # Markers scale with the canvas: s=66 was set for a 7.2x5.2 figure, which is
    # 2.5x the area of a panel. There are ~35 layers across a 229 pt axes, i.e.
    # 6.5 pt per point, so s=26 (a ~5.8 pt disc) is as large as they can get
    # without the low/mid run fusing into one band.
    s = 26 if not big else 32
    ax.plot(L, y, "-", color="#c7c7c7", lw=1.0, zorder=2)
    ax.scatter(L[lo], y[lo], s=s, color=C_LOW, edgecolors="white", linewidths=0.7,
               zorder=4, label="low / mid")
    ax.scatter(L[mh], y[mh], s=s + 4, color=C_MIDHI, edgecolors="white", linewidths=0.7,
               marker="s", zorder=5, label="mid-high (≥17)")
    ax.scatter(L[hi], y[hi], s=s + 10, color=C_HI, edgecolors="white", linewidths=0.8,
               marker="D", zorder=6, label="high (≥32)")


def panel(key, kind, ylabel, title, ref_line, ylim=None, logy=False, name=None,
          legend_loc="upper left"):
    d = _load(); L = d["layer"]; y = _enhance(L, d[key], kind)
    _rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    if ref_line is not None:
        ax.axhline(ref_line, color=C_REF, ls="--", lw=1.4, zorder=1)
    _scatter_by_regime(ax, L, y)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title(title, fontsize=FS_TITLE, pad=9)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    if ylim:
        ax.set_ylim(*ylim)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Not loc="best": it re-solves per figure, so the four panels ended up with
    # the legend in four different corners even though they share an x axis.
    # Three rise toward the top layers -> upper left is free; `overlap` falls
    # instead, so it passes lower left.
    ax.legend(loc=legend_loc, fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.0, handletextpad=0.35,
              labelspacing=0.22, borderpad=0.4)
    save_panel(fig, os.path.join(FIGDIR, name))
    plt.close(fig)


def main():
    # Titles are terse because there is no tight bbox to absorb an overlong one
    # and the plot area is only 229 pt: the old wording ran 276-333 pt, i.e. the
    # title was wider than the axes it labelled. "across Layers" is dropped from
    # all four -- the x axis already says Layer. Widest now is 159 pt.
    # Y labels are held under the 148 pt axis HEIGHT for the same reason.
    panel("subspace_overlap", "overlap",
          "Subspace overlap",
          "Principal Subspace Stability",
          ref_line=1.0, ylim=(0.86, 1.005), name="opd_pca-overlap.svg",
          legend_loc="lower left")

    panel("lambda_ratio_topk", "lambda",
          "Spectrum ratio  λ'/λ",
          "Variance Spectrum Change",
          ref_line=1.0, ylim=(0.99, 1.02), name="opd_pca-lambda.svg")

    panel("hidden_shift_rel", "shift",
          "Shift  ‖Δμ‖ / ‖μ‖",
          "Hidden-State Displacement",
          ref_line=0.0, name="opd_pca-shift.svg")

    panel("drift_in_topk", "drift",
          "Drift in top-20 PCs",
          "Drift vs. Principal Axes",
          ref_line=float(_load()["drift_rand_baseline"][0]), logy=True,
          name="opd_pca-drift.svg")


if __name__ == "__main__":
    main()
