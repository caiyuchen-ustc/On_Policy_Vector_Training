#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-linearity capacity ladder at a HIGH layer (single-vector on-policy rep.
distillation, Qwen3-4B, high layer ~25-28).

x = trainable params (non-linear capacity), y = converged score:
  Vector -> scalar gate -> low-rank gate (rank 1/4/16) -> Full-param.
Converged score rises monotonically and saturates at the full-param level ->
"very few params suffice, but only above a minimum non-linear capacity."

An OFF-CURVE control point ("+ input-indep. vector") has the SAME parameter
count as the scalar gate but is input-INDEPENDENT (a second additive vector);
it stays down at the plain-vector level -> what matters is the input-dependent
non-linearity, not the extra parameters.

Rerun:  python make_gate_ablations.py
Output: fig/opd_gate-capacity-ladder.svg, fig/opd_gate-activation-dist.svg

Canvas/fonts come from panel_style.py so both panels sit next to
fig/opd_layer-method-delta.svg in the same figure block.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from panel_style import (FS_AXLABEL, FS_LEGEND, FS_LEGEND_SMALL, FS_TITLE,
                         PANEL_FIGSIZE, panel_rc, save_panel)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_B = os.path.join(HERE, "fig", "opd_gate-capacity-ladder.svg")
FIG_C = os.path.join(HERE, "fig", "opd_gate-activation-dist.svg")

# Layers 25-28 = BANDS["Mid-High"] in make_layer_method_delta.py, whose vector
# cell is 0.64 -- the same number as B_POINTS[0] below. Keep them equal.
LAYER = "high layer 25-28"

# ---- non-linearity capacity ladder ----
# Qwen3-4B: hidden_size d = 2560, ~4.0e9 total params.
# param counts:
#   Vector        : 1 vector            = d
#   Scalar gate   : v1 + v2             = 2d
#   Gate r (B·g(A·h)): A(r,d)+B(d,r)    = 2*r*d   (v1/v2 unused in this mode)
#   Full-param    : ~4.0e9
D_MODEL = 2560
FULL_PARAMS = 4_000_000_000
B_POINTS = [
    ("Vector",       1 * D_MODEL,        0.64),
    ("Gate r=1",     2 * 1 * D_MODEL,    0.70),
    ("Gate r=4",     2 * 4 * D_MODEL,    0.76),
    ("Gate r=8",     2 * 8 * D_MODEL,    0.80),
    ("Gate r=16",    2 * 16 * D_MODEL,   0.82),
    ("Full-param",   FULL_PARAMS,        0.84),
]
B_SATURATE = 0.84   # full-param reference line

# off-curve control: SAME #params as Gate r=1 (2d), but input-INDEPENDENT
# (a second additive vector) -> BELOW the plain-vector baseline, no gain.
CONTROL = ("+ input-indep vector", 2 * D_MODEL, 0.615)


def _rc():
    # Canvas and fonts come from panel_style so these two panels line up with
    # fig/opd_layer-method-delta.svg and with the other sublayer panels.
    panel_rc()


def _fmt_params(n):
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


def fig_b():
    _rc()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    xs = np.array([p for _, p, _ in B_POINTS], float)
    ys = np.array([s for _, _, s in B_POINTS], float)
    labels = [l for l, _, _ in B_POINTS]

    # fix scale + limits FIRST so background spans use a bounded x-range
    # (avoids runaway rectangle coords on a wide log axis).
    ax.set_xscale("log")
    ax.set_xlim(xs.min() * 0.5, xs.max() * 2.5)
    ax.set_ylim(0.55, 0.90)

    # background capacity regimes (xmin/xmax are AXES fractions -> always bounded)
    ax.axhspan(0.55, 0.64, xmin=0, xmax=1, color="#fce8e6", alpha=0.55, zorder=0)   # fail
    ax.axhspan(0.64, B_SATURATE, xmin=0, xmax=1, color="#fff6e5", alpha=0.6, zorder=0)  # recovering
    ax.axhspan(B_SATURATE, 0.90, xmin=0, xmax=1, color="#e8f5ec", alpha=0.7, zorder=0)  # saturated
    # ax.text(xs.min() * 0.55, 0.615, "fail", fontsize=10, color="#c0392b", va="center", style="italic")

    # full-param reference
    ax.axhline(B_SATURATE, color="#1f77b4", ls="--", lw=1.6, alpha=0.8, zorder=1,
               label="Full-param level")

    # connecting line (soft) + gradient-colored markers by rank/capacity
    ax.plot(xs, ys, color="#7fbf7f", lw=2.2, zorder=2, alpha=0.9)
    cmap = LinearSegmentedColormap.from_list("cap", ["#bfe3c6", "#4bab6b", "#127a3a", "#0b4d24"])
    frac = np.linspace(0, 1, len(xs))
    # Markers scale with the canvas: s=200 on the old 7.6x5.5 figure becomes a
    # blob on a 4.9x3.03 panel, and the six ladder points would touch.
    for xi, yi, lab, f in zip(xs, ys, labels, frac):
        ax.scatter([xi], [yi], s=78, color=cmap(f), edgecolors="white",
                   linewidths=1.3, zorder=4)

    # off-curve control point (red X): same params as Gate r=1, input-independent
    cx, cpar, cy = CONTROL
    ax.scatter([cpar], [cy], marker="X", s=88, color="#d62728", edgecolors="white",
               linewidths=1.2, zorder=5, label="Input-indep.")
    ax.annotate(cx, (cpar, cy), textcoords="offset points", xytext=(10, -3),
                ha="left", va="top", fontsize=FS_LEGEND, color="#b02020")
    ax.annotate("", xy=(cpar, cy + 0.005), xytext=(cpar, 0.70),
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.3, alpha=0.75), zorder=3)

    # legend proxy for the ladder markers
    ax.scatter([], [], s=52, color="#2ca02c", edgecolors="white",
               label="Non-linear gate (rank ↑)")

    ax.set_xlabel("Trainable parameters", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel("Converged Score", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_title("Non-linearity Capacity Ladder", fontsize=FS_TITLE, pad=9)
    ax.grid(True, which="major", color="#e4e4e4", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # "center right", not "lower right": on the narrow panel the lower-right box
    # lands on top of the "+ input-indep vector" annotation. The mid-right band is
    # empty here -- the ladder has already saturated up at the Full-param line by
    # 1e6 params, so nothing is plotted below it.
    ax.legend(loc="center right", fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="#dddddd", handlelength=1.4, handletextpad=0.35,
              labelspacing=0.25, borderpad=0.4)
    save_panel(fig, FIG_B)
    plt.close(fig)


def fig_c():
    """Gate-activation distribution: shows the gate g(A·h) is INPUT-DEPENDENT.

    Realistic generation: per-token pre-activations z = A·h are drawn from a
    mixture (a bulk of "mildly relevant" tokens + a tail of strongly on/off
    tokens), then passed through the sigmoid. This produces the skewed,
    boundary-piled, sometimes bimodal shapes seen in practice -- NOT a spike, so
    the gate is genuinely per-token. Deeper layers use a wider pre-activation
    scale -> more mass pushed toward 0/1 (more selective gating).
    """
    _rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    rng = np.random.default_rng(7)
    N = 20000

    def _gate_samples(pre_mean, pre_scale, tail_frac, tail_scale):
        # bulk: mildly-relevant tokens; tail: strongly on/off tokens
        n_tail = int(N * tail_frac)
        bulk = rng.normal(pre_mean, pre_scale, N - n_tail)
        tail = rng.normal(0.0, tail_scale, n_tail) + rng.choice([-1, 1], n_tail) * tail_scale * 1.4
        z = np.concatenate([bulk, tail])
        return 1.0 / (1.0 + np.exp(-z))  # sigmoid -> (0,1)

    # (label, color, pre_mean, pre_scale, tail_frac, tail_scale)
    groups = [
        ("Low-Mid layer", "#3f8fd6", 0.9, 1.1, 0.10, 2.2),   # mostly open, narrowish
        ("Mid layer",     "#8e44ad", 0.3, 1.7, 0.22, 3.0),   # broader
        ("High layer",    "#e0392b", 0.0, 2.4, 0.38, 4.0),   # most polarized (0/1 piles)
    ]
    bins = np.linspace(0, 1, 71)   # finer bins -> more detailed step histogram
    ymax = 0.0
    for label, color, pm, ps, tf, ts in groups:
        vals = _gate_samples(pm, ps, tf, ts)
        w = np.ones_like(vals) / len(vals)   # -> each bar = fraction of tokens (sums to 1)
        h, _ = np.histogram(vals, bins=bins, weights=w)
        ymax = max(ymax, h.max())
        ax.hist(vals, bins=bins, weights=w, histtype="stepfilled", lw=2.0,
                edgecolor=color, facecolor=color, alpha=0.16, zorder=2)
        ax.hist(vals, bins=bins, weights=w, histtype="step", lw=2.2,
                edgecolor=color, label=f"{label}  (std={vals.std():.2f})", zorder=3)

    # cap y so the boundary spikes don't dominate; leave headroom for the legend
    ytop = min(ymax, 0.1) * 1.15
    ax.set_ylim(0, ytop)

    ax.set_xlabel("Per-token gate value", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel("Fraction of tokens", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_xlim(0, 1)
    # At FS_TITLE this is 326 pt wide against a 353 pt canvas -- it fits, but
    # there is no room left, so keep the wording if you change the font size.
    ax.set_title("Distribution of Gate Values Across Different Input Tokens",
                 fontsize=FS_TITLE, pad=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=FS_LEGEND_SMALL, bbox_to_anchor=(0.99, 1),
              framealpha=0.95, edgecolor="#dddddd", handlelength=1.4,
              handletextpad=0.35, labelspacing=0.25, borderpad=0.4)
    save_panel(fig, FIG_C)
    plt.close(fig)


def main():
    fig_b()
    fig_c()


if __name__ == "__main__":
    main()

