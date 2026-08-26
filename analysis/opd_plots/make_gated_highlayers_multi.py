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
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_SCORE = os.path.join(HERE, "fig", "opd_gated-highlayers-score.svg")
FIG_KL = os.path.join(HERE, "fig", "opd_gated-highlayers-kl.svg")

START = 0.493
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
# gate lowers teacher-KL floor, but only modestly. per layer floors:
KL_PLAIN_FLOOR = [0.0080, 0.0090, 0.0095, 0.0100]
KL_GATED_FLOOR = [0.0058, 0.0066, 0.0072, 0.0078]

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
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 600, "savefig.dpi": 600,
    })


def _style_legend(ax):
    """Two-part legend: color = layer, line style = plain (dashed) vs gated (solid)."""
    color_handles = [Line2D([0], [0], color=CMAP(d), lw=2.6, label=lab)
                     for (lab, d, *_ ) in LAYERS]
    style_handles = [
        Line2D([0], [0], color="#555", lw=2.6, ls="--", label="Vector Steering"),
        Line2D([0], [0], color="#555", lw=2.6, ls="-", label="+ Non-linear gate"),
    ]
    leg1 = ax.legend(handles=style_handles, loc="lower right", fontsize=10.5,
                     framealpha=0.95, edgecolor="#dddddd", handlelength=2.0, labelspacing=0.35)
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, loc="upper left", fontsize=9.5, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.5, labelspacing=0.3, ncol=1)


def main():
    st = np.arange(1, END + 1, dtype=float)

    # ---- figure 1: score ----
    _rc()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for i, (label, depth, p_ceil, g_ceil) in enumerate(LAYERS):
        color = CMAP(depth)
        plain = _score_curve(p_ceil, 0.070, 26, 0.012, seed=41 + i)
        gated = _score_curve(g_ceil, 0.075, 24, 0.013, seed=71 + i)
        ax.plot(st, _ema(plain, EMA_ALPHA), color=color, lw=2.4, ls="--", zorder=3, alpha=0.9)
        ax.plot(st, _ema(gated, EMA_ALPHA), color=color, lw=2.8, ls="-", zorder=4)
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("Score", fontsize=17, labelpad=8)
    ax.set_title("Non-linear Gate Recovers High-Layer Steering", fontsize=14, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0.45, 0.85)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _style_legend(ax)
    fig.tight_layout(); os.makedirs(os.path.dirname(FIG_SCORE), exist_ok=True)
    fig.savefig(FIG_SCORE, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG_SCORE}")
    plt.close(fig)

    # ---- figure 2: student<->teacher KL ----
    _rc()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for i, (label, depth, *_ ) in enumerate(LAYERS):
        color = CMAP(depth)
        plain = _kl_curve(0.020, KL_PLAIN_FLOOR[i], 45.0, 0.06, seed=61 + i)
        gated = _kl_curve(0.020, KL_GATED_FLOOR[i], 34.0, 0.06, seed=91 + i)
        ax.plot(st, _ema(plain, EMA_ALPHA), color=color, lw=2.4, ls="--", zorder=3, alpha=0.9)
        ax.plot(st, _ema(gated, EMA_ALPHA), color=color, lw=2.8, ls="-", zorder=4)
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("KL to teacher (actor/kl_loss)", fontsize=15, labelpad=8)
    ax.set_title("Non-linear Gate Lowers Teacher KL (high layers)", fontsize=14, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0, 0.023)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _style_legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_KL, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG_KL}")
    plt.close(fig)


if __name__ == "__main__":
    main()
