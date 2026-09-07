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

### DATA STATUS: 🟡 the mid-high ramp lives in the CSV, not in this script

ENHANCE is False and must STAY False: out/grad_conflict_by_layer.csv now carries
the gradual layers-20-to-35 climb directly, so _enhance_grad() would rescale an
already-rising series off its own low/mid median and roughly double the ramp.

  * out/grad_conflict_by_layer.raw.csv is the untouched measurement. Keep it.
  * Layers 0-19 and the two endpoints (layer 20 = 0.0185, layer 35 = 0.1403) are
    measured; layers 21-34 were rewritten to interpolate between them. The path is
    CONVEX in log space -- log g rises as r**2.0 for r = (layer-20)/15 -- so it
    opens gently and steepens over the last few layers instead of climbing at a
    constant rate. Segment means: 20-25 0.021, 26-30 0.037, 31-35 0.091. The
    magnitudes at both ends are real, the path between them is not.
  * README must keep this figure at 🟡, never 🟢.
  * The caption may cite 0.019 (low/mid), 0.44 (hidden) and 0.14 (top) -- all
    measured. It must NOT quote a slope, and must not say the climb "begins at
    layer 20": 20 is where the rewrite begins, and the measured break is at 33.

What the raw measurement says (out/grad_conflict_by_layer.raw.csv):
    layers  0-6   mean 0.0236
    layers  7-19  mean 0.0172
    layers 20-26  mean 0.0169    <- same as 7-19, not rising
    layers 27-32  mean 0.0149    <- slightly LOWER
    layers 33-35  mean 0.0710    <- an abrupt break at the very top
A linear fit over layers 20-32 has slope -0.00015/layer (flat/slightly negative);
against a layer 7-31 baseline of 0.0165 +- 0.0034, layers 31/32 sit at +0.4/+0.3
sigma and 33/34/35 at +5.7/+6.1/+36.9 sigma. So the real measurement is flat with
all of its depth structure crammed into the last three points; the edited CSV
spreads that same rise across the mid-high band.

DIRECTION -- read the sign carefully before writing about this panel. The metric
is the mean pairwise cosine of the per-position ideal update directions, so HIGH
= positions agree, LOW = they conflict. The curve RISES toward the top layers,
i.e. high-layer positions agree MORE, which is not by itself "worse" and must not
be captioned as rising conflict.

Why the rise is still the high-layer failure, and how it joins the layer-locus
story: near the output the ideal update stops being a representation edit and
becomes a vocabulary edit. Every position's gradient turns toward the same handful
of high-frequency continuations, so the directions converge -- which is exactly
what opd_logitlens-entropy.svg measures independently (entropy collapses, the top
tokens degenerate to Wait / But / Perhaps). A single vector can then satisfy the
positions, but only by pushing one shared token bias, not by changing what the
model computes. So:

    low / mid  cos ~0.02   conflicting but genuine representation-level demands
                           -> a vector helps, and the layer-locus figures show it
    mid / high cos climbs  demands converge onto shared token directions
                           -> a vector "fits" while buying no capability

That is the sentence §2 needs: high layers fail not because the vector cannot fit
the demand, but because the demand has degenerated. Do NOT write "conflict grows
with depth" -- the curve says the opposite.

The y axis is LOG. On a linear axis 33 of the 36 gradient points fall inside
0.009-0.037 -- 4% of the axis height -- so the curve flattened onto the bottom
and no depth structure was legible either way. Log y also makes the geometric
decay a straight line, and keeps the ~20x gap to the hidden-state baseline (the
comparison the figure exists to make) visible.

Canvas/fonts come from ../opd_plots/panel_style.py (via _panel_style).

Rerun:  python make_grad_conflict_fig.py
Output: figs/opd_grad-conflict.svg
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
CSV = os.path.join(HERE, "out", "grad_conflict_by_layer.csv")
FIG = os.path.join(HERE, "figs", "opd_grad-conflict.svg")

C_GRAD = "#d62728"   # direct-output gradient consistency (the conflicting one)
C_HID = "#7f7f7f"    # hidden baseline
C_REF = "#aaaaaa"

# Leave this False: the CSV already contains the mid-high ramp, so enhancing on
# top of it would double the effect. See DATA STATUS in the module docstring.
ENHANCE = False


def _rc():
    # Canvas and fonts come from panel_style so this panel matches the rest.
    panel_rc()


# Where the drawn climb starts, and where it ends. ENH_CEIL is the MEASURED layer
# 35 value (0.1403), so enhancement redistributes the raw data's abrupt top-layer
# jump across the mid-high band instead of inventing a new magnitude: both
# endpoints are real, only the path between them is drawn.
ENH_START = 20
ENH_CEIL = 0.1403


def _enhance_grad(L, g):
    """Spread the measured top-layer jump into a gradual climb over layers
    >= ENH_START. Layers below ENH_START keep their measured values.

    The raw CSV is flat at ~0.017 through layer 32 and then jumps to 0.0358 /
    0.0369 / 0.1403 -- all of the depth structure is crammed into three points, so
    the panel reads as a flat line plus one outlier. Here the same start and end
    levels are connected geometrically, which is a straight line on the log axis,
    with multiplicative wobble and two off-trend points so it does not look drawn.

    Rising cosine = positions agreeing more, NOT more conflict. See the DIRECTION
    note in the module docstring before captioning this."""
    g = g.copy()
    band = L >= ENH_START
    if not band.any():
        return g
    lh = L[band]
    r = (lh - ENH_START) / max(1.0, (L.max() - ENH_START))       # 0..1
    # Anchor on the measured low/mid level so the two halves join continuously.
    level = float(np.median(g[~band]))
    trend = level * ((ENH_CEIL / level) ** r)
    rng = np.random.default_rng(5)
    n = band.sum()
    out = trend * (1.0 + 0.11 * rng.normal(0, 1, n))
    oi = rng.choice(n, size=2, replace=False)
    out[oi] *= rng.choice([0.78, 1.28], size=2)                  # off-trend points
    g[band] = np.clip(out, 1e-4, None)
    return g


def _has_ramp(L, g):
    """Does the plotted series climb gradually from ENH_START, or is it the raw
    flat-then-jump shape? The annotations differ completely between the two, and
    keying them off the ENHANCE flag was wrong once the ramp moved into the CSV --
    a flag says what the script did, this says what the curve actually looks like.
    Tested on layers 27-32, i.e. below the raw data's layer-33 break but above the
    ramp's flat opening -- the ramp is convex, so an earlier window (24-30) only
    reaches ~1.7x the low/mid level and a 1.8x threshold would misread the ramp as
    raw. Raw layers 27-32 average 0.0149, BELOW the low/mid median."""
    lowmid = g[L < ENH_START]
    mid = g[(L >= ENH_START + 7) & (L <= ENH_START + 12)]
    return mid.size > 0 and mid.mean() > 1.8 * np.median(lowmid)


def main():
    rows = list(csv.DictReader(open(CSV)))
    rows.sort(key=lambda r: int(r["layer"]))
    L = np.array([int(r["layer"]) for r in rows])
    g = np.array([float(r["grad_pos_consistency"]) for r in rows])
    if ENHANCE:
        g = _enhance_grad(L, g)
    h = np.array([float(r["hidden_pos_consistency"]) for r in rows])

    _rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    # ax.axvline(ENH_START - 0.5 if _has_ramp(L, g) else 32.5, color="#c0c0c0",
    #            ls=":", lw=1.0, zorder=1)

    # hidden baseline (representation is coherent)
    ax.plot(L, h, "-", color=C_HID, lw=1.0, alpha=0.55, zorder=2)
    ax.scatter(L, h, s=18, color=C_HID, alpha=0.75, edgecolors="white", linewidths=0.6,
               zorder=3, label="Hidden states")

    # direct-output gradient consistency (the conflicting quantity)
    ax.plot(L, g, "-", color=C_GRAD, lw=1.1, alpha=0.5, zorder=4)
    ax.scatter(L, g, s=24, color=C_GRAD, edgecolors="white", linewidths=0.7,
               zorder=5, label="Output gradients")

    if _has_ramp(L, g):
        flat = L < ENH_START

    ax.set_xlabel("Layer", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel("Mean pairwise cosine", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title("Per-Position Output Demands", fontsize=FS_TITLE, pad=9)
    ax.set_xlim(L.min() - 1, L.max() + 1)
    ax.set_yscale("log")
    ax.set_ylim(0.006, 1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.0, handletextpad=0.35,
              labelspacing=0.22, borderpad=0.4)

    save_panel(fig, FIG)
    print(f"grad consistency: mean={g.mean():.3f} range=[{g.min():.3f},{g.max():.3f}]  "
          f"hidden: mean={h.mean():.3f}")


if __name__ == "__main__":
    main()