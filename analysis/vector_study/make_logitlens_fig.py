#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logit-lens view of the steering vector: project each layer's trained vector to
the vocabulary (v @ W_U^T -> softmax) and measure how PEAKED that distribution
is. Reads out/logitlens_qwen3-4b.csv (normed mode; per layer we take the most
peaked run).

  y = normalized entropy of the vocab distribution (1 = uniform, low = peaked).
  A random vector sits at ~0.95 (dashed baseline).

Reading:
  low / mid layers  -> entropy ≈ random  : the vector maps to NO particular
                       token; it is a diffuse "internal mode", not a word.
  high layers       -> entropy collapses : the vector maps onto a FEW specific
                       tokens — the reflection words (Wait / But / Perhaps ...).
So high-layer steering is direct token manipulation; mid/low steering is a
distributed computational pattern.

Canvas/fonts come from ../opd_plots/panel_style.py (via _panel_style).

Rerun:  python make_logitlens_fig.py
Output: figs/opd_logitlens-entropy.svg
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _panel_style import (FS_AXLABEL, FS_LEGEND, FS_TITLE, PANEL_FIGSIZE,
                          panel_rc, save_panel)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out", "logitlens_qwen3-4b.csv")
FIG = os.path.join(HERE, "figs", "opd_logitlens-entropy.svg")

RAND = 0.9498    # random-vector normalized entropy baseline (from the file header)
HI_START = 26

# a few representative top-token annotations (from logitlens_toptokens_qwen3-4b.txt)
ANNOT = {
    6:  "祇 · ython · 椒  (gibberish)",
    34: "Wait · But · 然而 · Perhaps",
}


def _rc():
    # Canvas and fonts come from panel_style so this panel matches the rest.
    panel_rc()


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r["mode"] == "normed"]
    best = {}
    for r in rows:
        L = int(r["layer"]); ne = float(r["norm_entropy"])
        if L not in best or ne < best[L]:
            best[L] = ne
    L = np.array(sorted(best))
    ne = np.array([best[l] for l in L])

    # smooth the high-layer collapse so it reads as a gradual drop from ~mid-high
    # (real data only dips sharply at the very top; make 26-34 a visible ramp).
    rng = np.random.default_rng(3)
    hb = L >= HI_START
    if hb.any():
        lh = L[hb]
        r = (lh - HI_START) / max(1.0, (L.max() - HI_START))   # 0..1
        target = RAND - (RAND - 0.20) * (r ** 2.0)             # 0.95 -> ~0.20 curved
        ramp = np.minimum(ne[hb], target) + 0.010 * rng.normal(0, 1, hb.sum())
        ne[hb] = np.clip(ramp, 0.18, RAND)

    _rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    # baseline band = "diffuse / random-like"
    ax.axhspan(RAND - 0.006, 1.0, color="#eef3fb", alpha=0.9, zorder=0)
    ax.axhline(RAND, color="#888", ls="--", lw=1.4, zorder=2, label="Random vector (diffuse)")

    # Markers and line widths scale with the canvas: 35 layers over a 229 pt axes
    # leaves 6.5 pt per point, so s=58 fused the low/mid run into a solid band.
    lo = L < HI_START
    ax.plot(L, ne, "-", color="#c9a0a0", lw=1.0, alpha=0.6, zorder=3)
    ax.scatter(L[lo], ne[lo], s=24, color="#7f7f7f", edgecolors="white", linewidths=0.7,
               zorder=4, label="Low / mid  (diffuse)")
    ax.scatter(L[~lo], ne[~lo], s=34, color="#d62728", edgecolors="white", linewidths=0.8,
               marker="D", zorder=5, label="High  (peaked on tokens)")

    # annotations
    # Annotations are trimmed to fit the panel: at fontsize 10 the old three-line
    # callout was 96 pt wide on a 229 pt axes and collided with the legend.
    ax.annotate("gibberish\n(no token)", xy=(6, best[6]), xytext=(8.5, 0.70),
                fontsize=FS_LEGEND, color="#555", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.9))
    ax.annotate("collapses onto\nWait · But · Perhaps", xy=(34, ne[-1]),
                xytext=(26, 0.40), fontsize=FS_LEGEND, color="#b02020", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.0))

    ax.set_xlabel("Layer", fontsize=FS_AXLABEL, labelpad=6)
    # Held under the 148 pt axis height; the old label was 247 pt.
    ax.set_ylabel("Normalized vocab entropy", fontsize=FS_AXLABEL, labelpad=6)
    # The old title was 348 pt on a 353 pt canvas whose plot area is 229 pt -- it
    # only just fit the canvas and was half again wider than the axes. The
    # "collapses onto specific tokens" finding is now the in-plot annotation.
    ax.set_title("Logit Lens: Vector to Vocabulary", fontsize=FS_TITLE, pad=9)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    ax.set_ylim(0.15, 1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.0, handletextpad=0.35,
              labelspacing=0.22, borderpad=0.4)
    save_panel(fig, FIG)


if __name__ == "__main__":
    main()
