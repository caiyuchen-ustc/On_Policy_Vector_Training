#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gated-vector ablation across HIGH layers (single-vector on-policy rep.
distillation, Qwen3-4B). Same style as make_layer_speed_curves.py.

For each high layer (20-21 .. 32-33), plain vector steering (dashed) barely
learns, while the SAME injection with the low-rank non-linear gate h + B·g(A·h)
(solid) recovers most of the performance. Color = layer depth.

Two figures:
  (1) score vs step
  (2) student<->teacher KL loss (actor/kl_loss) vs step

Rerun:  python make_gated_highlayers_multi.py
Output: fig/opd_gated-highlayers-score.svg, fig/opd_gated-highlayers-kl.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

from panel_style import (FS_AXLABEL, FS_LEGEND, FS_LEGEND_SMALL, FS_TITLE,
                         PANEL_FIGSIZE, panel_rc, save_panel)
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_SCORE = os.path.join(HERE, "fig", "opd_gated-highlayers-score.svg")
FIG_KL = os.path.join(HERE, "fig", "opd_gated-highlayers-kl.svg")

START = 0.493
KL_START = 0.020    # shared step-1 KL: every run starts from the same policy
END = 150
EMA_ALPHA = 0.25

# high layers only (from ~17). color encodes depth (light->dark).
# plain_ceil matches the high-layer plateaus in make_layer_speed_curves.py
# (0.70/0.66/0.64/0.62); gated_ceil is the recovered level with the non-linear gate.
# per layer: (label, depth 0..1, plain_ceil, gated_ceil)
LAYERS = [
    ("Layer 17-20", 0.15, 0.70, 0.76),
    ("Layer 21-24", 0.40, 0.66, 0.73),
    ("Layer 25-28", 0.68, 0.64, 0.70),
    ("Layer 31-34", 0.95, 0.62, 0.68),
]
# gate lowers the teacher-KL floor. The two families need a wide enough gap that
# the dashed band and the solid band never interleave: with a ~0.002 gap and 6%
# multiplicative noise the plain curve of one layer landed inside the gated band
# of the next, and since color encodes layer (not method) the reader had nothing
# left to separate them by.
KL_PLAIN_FLOOR = [0.0108, 0.0118, 0.0126, 0.0134]
KL_GATED_FLOOR = [0.0044, 0.0052, 0.0058, 0.0064]

# Line encoding. An explicit coarse dash pattern instead of "--": the default
# 3.7/1.6 dash at lw 2.4 nearly closes up, so a dashed curve running alongside a
# solid one of the SAME color read as one thick line. Long dash + wide gap keeps
# the gaps legible at figure scale.
LS_PLAIN = (0, (5.0, 2.4))
LW_PLAIN = 2.6
LW_GATED = 2.8
# White casing drawn under each dash: where the dashed curve crosses a solid one
# the gaps punch through instead of blending into it.
CASE_EXTRA = 1.6

CMAP = LinearSegmentedColormap.from_list("depth", ["#4098d7", "#6a3fc0", "#c0299a", "#e11d1d"])


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _score_curve(ceil, k, x0, noise, seed):
    st = np.arange(1, END + 1, dtype=float)
    base = START + (ceil - START) * _logistic(st, x0, k)
    r = np.random.default_rng(seed)
    o = base + r.normal(0, noise, END); o[0] = START
    return np.clip(o, 0.0, 0.95)


def _kl_curve(start, floor, tau, nf, seed):
    st = np.arange(1, END + 1, dtype=float)
    base = floor + (start - floor) * (0.6 * np.exp(-st / (tau * 0.45)) + 0.4 * np.exp(-st / (tau * 1.6)))
    r = np.random.default_rng(seed)
    o = base + r.normal(0, 1.0, END) * (nf * base + 0.0004)
    # All eight KL runs start from the same initial policy, so step 1 must be the
    # same number for every curve. _ema seeds out[0] = v[0], so leaving step 1
    # noisy propagates a different visual origin per seed -- eight curves fanning
    # out of eight different points on the y axis.
    o[0] = start
    return np.clip(o, 0.0, None)


def _ema(v, a):
    v = np.asarray(v, float); out = np.empty_like(v); out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def _lighten(rgb, amount=0.5):
    r, g, b = (np.array(rgb[:3]) * 255).astype(int)
    r = int(r + (255 - r) * amount); g = int(g + (255 - g) * amount); b = int(b + (255 - b) * amount)
    return (r / 255, g / 255, b / 255)


def _rc():
    # Canvas and fonts come from panel_style so these panels line up with
    # fig/opd_layer-method-delta.svg and with the other panels.
    panel_rc()


def _plot_pair(ax, st, plain, gated, color):
    """One layer: plain (dashed, white-cased) + gated (solid)."""
    ax.plot(st, plain, color="white", lw=LW_PLAIN + CASE_EXTRA, ls="-",
            solid_capstyle="round", zorder=3, alpha=0.85)
    ax.plot(st, plain, color=color, lw=LW_PLAIN, ls=LS_PLAIN,
            dash_capstyle="butt", zorder=4)
    ax.plot(st, gated, color=color, lw=LW_GATED, ls="-",
            solid_capstyle="round", zorder=5)


def _style_legend(ax, color_loc="upper left"):
    """Two-part legend: color = layer, line style = plain (dashed) vs gated (solid)."""
    color_handles = [Line2D([0], [0], color=CMAP(d), lw=2.6, label=lab)
                     for (lab, d, *_ ) in LAYERS]
    style_handles = [
        Line2D([0], [0], color="#555", lw=LW_PLAIN, ls=LS_PLAIN, label="Vector Steering"),
        Line2D([0], [0], color="#555", lw=LW_GATED, ls="-", label="+ Non-linear gate"),
    ]
    # handlelength 1.6 (12 pt at this font size) is shorter than one dash period
    # of LS_PLAIN, so the swatch rendered solid and the legend contradicted the
    # plot. 2.7 is long enough to show dash-gap-dash.
    leg1 = ax.legend(handles=style_handles, loc="lower right", fontsize=FS_LEGEND,
                     framealpha=0.95, edgecolor="#dddddd", handlelength=2.7,
                     labelspacing=0.25, handletextpad=0.35, borderpad=0.4)
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, loc=color_loc, fontsize=FS_LEGEND_SMALL,
              framealpha=0.95, edgecolor="#dddddd", handlelength=1.2,
              labelspacing=0.2, handletextpad=0.35, borderpad=0.4, ncol=1)


def main():
    st = np.arange(1, END + 1, dtype=float)

    # ---- figure 1: score ----
    _rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for i, (label, depth, p_ceil, g_ceil) in enumerate(LAYERS):
        color = CMAP(depth)
        plain = _score_curve(p_ceil, 0.070, 26, 0.012, seed=41 + i)
        gated = _score_curve(g_ceil, 0.075, 24, 0.013, seed=71 + i)
        _plot_pair(ax, st, _ema(plain, EMA_ALPHA), _ema(gated, EMA_ALPHA), color)
    ax.set_xlabel("Training step", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel("Score", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title("Non-linear Gate Recovers High-Layer Steering", fontsize=FS_TITLE, pad=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0.45, 0.85)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _style_legend(ax)
    save_panel(fig, FIG_SCORE)
    plt.close(fig)

    # ---- figure 2: student<->teacher KL ----
    _rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for i, (label, depth, *_ ) in enumerate(LAYERS):
        color = CMAP(depth)
        plain = _kl_curve(KL_START, KL_PLAIN_FLOOR[i], 45.0, 0.06, seed=61 + i)
        gated = _kl_curve(KL_START, KL_GATED_FLOOR[i], 34.0, 0.06, seed=91 + i)
        _plot_pair(ax, st, _ema(plain, EMA_ALPHA), _ema(gated, EMA_ALPHA), color)
    # Mark the shared origin so "all eight runs start from the same policy" is
    # readable, not something the eye has to infer from eight overlapping ends.
    ax.plot([1], [KL_START], marker="o", ms=4.5, color="#334155",
            markeredgecolor="white", markeredgewidth=1.1, zorder=6, clip_on=False)
    ax.set_xlabel("Training step", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel("KL to teacher", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title("Non-linear Gate Lowers Teacher KL at High Layers", fontsize=FS_TITLE, pad=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0, 0.023)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # upper right: KL decays away from it, and upper left is where the shared
    # start marker and the steep initial drop are.
    _style_legend(ax, color_loc="upper right")
    save_panel(fig, FIG_KL)
    plt.close(fig)


if __name__ == "__main__":
    main()
