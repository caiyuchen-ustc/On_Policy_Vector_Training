#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method (x, categorical) vs. layer depth (y, bottom->high) scatter; marker SIZE =
converged performance (RL-score recovery). Color = method.

X columns : Vector Steering | LoRA-4 | LoRA-8 | LoRA-16 | LoRA-32 | Full-param
Y axis    : bottom .. high  (only 3 band labels, no numeric ticks)

Claim: Vector steering is a single additive vector (no input-dependent
non-linearity), so it must borrow the downstream non-linear blocks -> works at
low/mid layers but FAILS at high layers (small dots at the top of the vector
column). LoRA / Full carry their own input-dependent capacity -> large dots at
every depth. => RL can be contained in very few params, but only above a
minimum non-linear capacity; not *any* few / *any* position.

Edit CONFIG (col, y, perf) and rerun:
    python make_layer_method_scatter.py
Output: fig/opd_layer-method-scatter.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_layer-method-scatter.svg")

TITLE = "Where RL Can Be Contained: Depth vs. Parameterization"
# metric behind "converged score" -- shown as a subtitle so the axis is unambiguous.
METRIC = "MATH500 accuracy @ convergence"

# x columns (order = plot order left->right): many params -> few params
COLS = ["Full-param", "LoRA-16", "LoRA-8", "Vector Steering"]
# single unified color; lighter = lower perf, darker = higher perf.
BASE_COLOR = "#f97316"          # coral orange
# y band labels (bottom .. high) shown at these y positions; no numeric ticks.
Y_BANDS = {0.04: "Bottom", 0.28: "Low-Mid", 0.5: "Mid", 0.72: "Mid-High", 0.96: "High"}

# marker size = perf (area). big dot = good, small dot = failed.
SIZE_MIN, SIZE_MAX = 22, 560
PERF_LO, PERF_HI = 0.15, 0.90

# ----------------------------------------------------------------------------
# Data.
#   Vector Steering: explicit points across depth (col, y in [0,1], perf);
#     only the top ones fail (small/light dots).
#   LoRA-4 / LoRA-16 / Full-param: params are enough, so LAYER PLACEMENT IS
#     FREE -> scatter each column's points across many depths (all high perf),
#     conveying "with enough params, any layer works". y from a fixed seed.
#   >>> replace numbers with your real sweep <<<
# ----------------------------------------------------------------------------
VECTOR_CONFIG = [
    ("Vector Steering", 0.03, 0.86),
    ("Vector Steering", 0.16, 0.86),
    ("Vector Steering", 0.29, 0.85),
    ("Vector Steering", 0.42, 0.82),
    ("Vector Steering", 0.55, 0.74),
    ("Vector Steering", 0.68, 0.60),
    ("Vector Steering", 0.81, 0.42),
    ("Vector Steering", 0.94, 0.23),   # highest layer -> tiny/light dot (fails)
]
# free columns: explicit points per column -> (col, [(y, perf), ...]).
# y uses the Y_BANDS positions: Bottom .04, Low-Mid .28, Mid .5, Mid-High .72, High .96
FREE_POINTS = [
    ("Full-param", [(0.04, 0.89), (0.50, 0.90), (0.96, 0.89)]),                 # Bottom / Mid / High
    ("LoRA-16",    [(0.28, 0.85), (0.72, 0.86), (0.96, 0.86)]),                 # Low-Mid / Mid-High / High
    ("LoRA-8",     [(0.16, 0.84), (0.50, 0.84), (0.96, 0.85)]),                 # Bottom~Low-Mid / Mid / High
]
FREE_SEED = 7                      # fixed layout seed (reproducible)
FREE_Y_MARGIN = 0.05               # keep random y within [margin, 1-margin]


def _build_config():
    cfg = list(VECTOR_CONFIG)
    for col, pts in FREE_POINTS:
        for y, perf in pts:
            cfg.append((col, float(y), float(perf)))
    return cfg


CONFIG = _build_config()


def _perf_to_size(p):
    frac = np.clip((p - PERF_LO) / (PERF_HI - PERF_LO), 0.0, 1.0)
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * frac


def _perf_to_color(p):
    """Lower perf -> lighter (toward white); higher perf -> full BASE_COLOR."""
    frac = np.clip((p - PERF_LO) / (PERF_HI - PERF_LO), 0.0, 1.0)
    h = BASE_COLOR.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    amt = 0.82 * (1.0 - frac)   # frac=1 -> base, frac=0 -> 82% toward white
    r = int(r + (255 - r) * amt); g = int(g + (255 - g) * amt); b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.labelsize": 12.5, "ytick.labelsize": 14,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(5.2, 6.6))
    xpos = {c: i for i, c in enumerate(COLS)}
    n = len(COLS)

    # very faint horizontal guide lines at each depth band (helps cross-column
    # comparison at the same depth without cluttering).
    for yb in Y_BANDS:
        ax.axhline(yb, color="#f0f0f0", lw=1.0, zorder=0)

    # draw dots: soft halo underneath + crisp dot on top.
    # all points in a column share the same x (column center); depth = y.
    # color = perf (lighter when smaller/worse), size = perf.
    for col, y, perf in CONFIG:
        s = _perf_to_size(perf)
        c = _perf_to_color(perf)
        xx = xpos[col]
        ax.scatter(xx, y, s=s * 1.7, c=c, alpha=0.14, edgecolors="none", zorder=2)
        ax.scatter(xx, y, s=s, c=c, alpha=0.95, edgecolors="white",
                   linewidths=1.3, zorder=3)

    ax.set_xticks(range(n))
    ax.set_xticklabels(COLS, rotation=16, ha="right")
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_xlabel("Parameterization  (many  →  few parameters)", fontsize=16, labelpad=8)

    ax.set_yticks(list(Y_BANDS.keys()))
    ax.set_yticklabels(list(Y_BANDS.values()))
    ax.set_ylim(-0.12, 1.12)
    ax.set_ylabel("Injected / trained depth", fontsize=17, labelpad=8)
    ax.tick_params(axis="y", length=5, width=1.0, color="#666666")
    ax.tick_params(axis="x", length=5, width=1.0, color="#666666")

    ax.set_title(TITLE, fontsize=15, pad=12)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # arrowheads on the x/y axes
    ax.plot(1, -0.12, ">", transform=ax.get_yaxis_transform(), clip_on=False,
            color="#666666", markersize=9)
    ax.plot(-0.65, 1, "^", transform=ax.get_xaxis_transform(), clip_on=False,
            color="#666666", markersize=9)

    # size legend: reference bubbles mapping marker size -> score.
    from matplotlib.lines import Line2D
    ref_scores = [0.9, 0.6, 0.3]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="", markeredgecolor="white",
               markerfacecolor=_perf_to_color(p), markeredgewidth=1.1,
               markersize=np.sqrt(_perf_to_size(p)), label=f"{p:.1f}")
        for p in ref_scores
    ]
    leg = ax.legend(handles=size_handles, loc="lower left",
                    bbox_to_anchor=(1.01, 0.0), frameon=True, framealpha=0.92,
                    edgecolor="#dddddd", fontsize=10.5, title="Score", title_fontsize=11,
                    labelspacing=1.7, borderpad=0.9, handletextpad=1.3)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
