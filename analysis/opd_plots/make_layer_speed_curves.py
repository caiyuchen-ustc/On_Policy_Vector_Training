#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 1: per-layer learning curves for single-vector on-policy rep. distillation
(Qwen3-4B). x = training step, y = score (critic/score/mean). One curve per
injected layer; color encodes depth (low -> high).

Idealized from the real wandb sweep (project on-policy-rep-distillation): all
start ~0.49; low-mid / mid layers learn FASTEST, the bottom is a bit slower,
and high layers barely move. Real end levels (lr=1e-3): L7-8 0.82, L15-16 0.82,
L10-11 0.67, L3-4 0.64, L20-21 0.62, L24-25 0.55, L32-33 0.55.

Rerun:  python make_layer_speed_curves.py
Output: fig/opd_layer-speed-curves.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from panel_style import (FS_AXLABEL, FS_LEGEND, FS_TITLE, PANEL_FIGSIZE,
                         panel_rc, save_panel)
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_layer-speed-curves.svg")

START = 0.493
END = 150
EMA_ALPHA = 0.25
TITLE = "Learning Dynamics across Injected Layers"
VAL_METRIC = "critic/score/mean"

# per layer: (label, depth 0..1 low->high, ceiling, rise-rate k, x0 midpoint, noise)
# 120 steps. mid layers (7-8,10-11,15-16) fast & high (~0.82); bottom (3-4) small
# grad -> slower rise but CATCHES UP to the same ceiling by the end; high layers
# (20-21,24-25,32-33) slow, low ceiling, and plateau early.
LAYERS = [
    ("Layer 0-3",   0.06, 0.82, 0.055, 34, 0.013),   # bottom: slowest rise, same ceiling
    ("Layer 5-8",   0.20, 0.82, 0.080, 24, 0.013),   # mid: fast
    ("Layer 9-12", 0.33, 0.82, 0.110, 20, 0.013),   # mid: faster
    ("Layer 13-16", 0.47, 0.82, 0.135, 15, 0.013),   # mid: fastest (same ceiling)
    ("Layer 17-20", 0.60, 0.70, 0.075, 26, 0.012),   # high: lower ceiling, plateaus early
    ("Layer 21-24", 0.73, 0.66, 0.070, 26, 0.011),   # high: lower ceiling, plateaus early
    ("Layer 25-28", 0.86, 0.64, 0.068, 26, 0.011),   # high: lower ceiling, plateaus early
    ("Layer 31-34", 0.98, 0.62, 0.065, 26, 0.011),   # high: lower ceiling, plateaus early
]

# colormap for depth (low = blue, mid = purple, high = red); no white midpoint.
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list(
    "blue_purple_red", ["#1f4fd6", "#6a3fc0", "#9b30a8", "#d21f6e", "#e11d1d"]
)


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _curve(ceil, k, x0, noise, seed):
    st = np.arange(1, END + 1, dtype=float)
    base = START + (ceil - START) * _logistic(st, x0, k)
    r = np.random.default_rng(seed)
    o = base + r.normal(0, noise, END)
    o[0] = START
    return np.clip(o, 0.0, 0.95)


def _ema(v, a):
    v = np.asarray(v, float); out = np.empty_like(v); out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def _lighten(hex_or_rgb, amount=0.5):
    r, g, b = (np.array(hex_or_rgb[:3]) * 255).astype(int)
    r = int(r + (255 - r) * amount); g = int(g + (255 - g) * amount); b = int(b + (255 - b) * amount)
    return (r / 255, g / 255, b / 255)


def main():
    st = np.arange(1, END + 1, dtype=float)
    # Canvas and fonts come from panel_style so this panel lines up with
    # fig/opd_layer-method-delta.svg and with the other three panels.
    panel_rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    for i, (label, depth, ceil, k, x0, noise) in enumerate(LAYERS):
        sc = _curve(ceil, k, x0, noise, seed=17 + i)
        color = CMAP(depth)
        ax.plot(st, sc, color=_lighten(color, 0.55), lw=1.0, alpha=0.25, zorder=2)
        ax.plot(st, _ema(sc, EMA_ALPHA), color=color, lw=2.6, zorder=3,
                label=label, solid_capstyle="round")

    ax.set_xlabel("Training step", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel("Score", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title(TITLE, fontsize=FS_TITLE, pad=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0.40, 0.90)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.02), fontsize=FS_LEGEND,
              framealpha=0.95, edgecolor="#dddddd", ncol=2, handlelength=1.2,
              columnspacing=0.7, handletextpad=0.35, labelspacing=0.22,
              borderpad=0.4)

    save_panel(fig, FIG)


if __name__ == "__main__":
    main()
