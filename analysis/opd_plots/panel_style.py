#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared canvas + font spec for every panel-sized figure in this folder.

Sublayer panels (placed next to fig/opd_layer-method-delta.svg):
    make_layer_speed_curves.py      -> opd_layer-speed-curves.svg
    make_layer_gradient_curves.py   -> opd_layer-gradient-curves.svg
    make_gated_highlayers_multi.py  -> opd_gated-highlayers-{score,kl}.svg
    make_gate_ablations.py          -> opd_gate-{capacity-ladder,activation-dist}.svg

Training-curve panels (see curve_panel_rc / style_curve_axes / curve_legend):
    plot_opd_score_curves.py        -> opd_<model>_ema0.2.svg   (4 on-policy)
    make_aime4b_data.py             -> opd_aime24-4b-offpolicy.svg
    make_code8b_data.py             -> opd_livecodebench-8b-offpolicy.svg
    make_ifevalg_data.py            -> opd_ifevalg-offpolicy.svg
    make_sciknow1p5b_data.py        -> opd_sciknoweval-1p5b-offpolicy.svg

Vector-direction panels in ../vector_study (they import this via
vector_study/_panel_style.py and write to vector_study/figs):
    make_injected_pca_figs2.py      -> opd_pca-{overlap,lambda,shift,drift}.svg
    make_logitlens_fig.py           -> opd_logitlens-entropy.svg
    make_hidden_subspace_fig.py     -> opd_hidden-subspace.svg   (1x2, wide row)

Why a shared module: the panels only line up with the delta heatmap if they
agree on the canvas size to the point AND on the font sizes. Keeping the
numbers in one place is the only way that survives editing one script.

Titles: the canvas is 352.8 pt wide and there is no tight bbox to absorb an
overlong title, so measure before you lengthen one -- at FS_TITLE roughly 55
characters fit, at FS_CURVE_TITLE roughly 66. Note the plot itself is only
229 pt wide (see PANEL_AXES_RECT), so a title near the canvas limit is visibly
wider than the axes it labels.

Geometry
--------
opd_layer-method-delta.svg is 392.944 x 436.242 pt. PANEL_H is exactly half its
height, so two panels stacked in a column are as tall as the heatmap and a
2-column x 2-row block of panels aligns with it top and bottom.

Do NOT save these panels with bbox_inches="tight": a tight bbox crops to the
drawn content, so a longer title or a wider tick label silently changes the
canvas size and the alignment is gone. Use save_panel() below, which pins the
axes rectangle instead.

Fonts
-----
Sizes are chosen so that the panels and the heatmap show the SAME apparent font
size when each is included at the same width in LaTeX. The heatmap is 5.457 in
wide with a 12.5 pt title; a panel is 4.9 in wide, i.e. it gets scaled up by
5.457/4.9 = 1.114, so 11.5 pt here renders as 12.8 pt -- within 2.5% of the
heatmap. Same reasoning puts the tick labels at 10 pt against the heatmap's 11.
"""

import os

import matplotlib.pyplot as plt

# Canvas: 4.9 x 3.029459 in = 352.8 x 218.121 pt.
DELTA_HEIGHT_PT = 436.242047       # measured from fig/opd_layer-method-delta.svg
PANEL_W = 4.9
PANEL_H = DELTA_HEIGHT_PT / 2 / 72
PANEL_FIGSIZE = (PANEL_W, PANEL_H)

# Symmetric left/right (right = 1 - left) so set_title, which centers on the
# AXES, also lands on the canvas center: the title sits directly above the plot
# and stays inside the canvas. An asymmetric rect forces a choice between those
# two. left=0.175 is set by the widest case in this group -- the KL panel, whose
# "0.000"-style ticks plus y label ran off the canvas at 0.145.
PANEL_AXES_RECT = {"left": 0.175, "right": 0.825, "top": 0.888, "bottom": 0.208}

FS_TITLE = 11.5
FS_AXLABEL = 12.5
FS_TICK = 10
FS_LEGEND = 7.5
FS_LEGEND_SMALL = 7.0


def panel_rc():
    """rcParams shared by all four panels."""
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "figure.dpi": 600, "savefig.dpi": 600,
    })


def save_panel(fig, path, dpi=600):
    """Pin the axes rect and write the SVG at exactly PANEL_FIGSIZE."""
    fig.subplots_adjust(**PANEL_AXES_RECT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg", dpi=dpi)
    print(f"[ok] saved: {path}")


# ---------------------------------------------------------------------------
# N panels on one row (a 1xN figure n times the width of a single panel).
#
# The point is that every subplot gets EXACTLY the same 229.3 x 148.3 pt axes as
# a standalone panel, so a row and a single panel show the same apparent font
# size and the same data density -- a 1x3 row is three panels side by side, not
# three shrunken ones. Write A for the panel axes width and m for the panel side
# margin; make the inter-panel gap 2m so an inner subplot's y label gets exactly
# the room the left one does. Then
#   2m + n*A + (n-1)*2m = n*(A + 2m) = n * PANEL_W * 72,
# which is an identity -- so the same A and m work for EVERY n, and
#   left = m / (n * PANEL_W * 72) = 0.175 / n,   wspace = 2m / A = 0.35 / 0.65.
# wspace is independent of n because subplots_adjust measures it in axes widths.
# ---------------------------------------------------------------------------
_WSPACE = (2 * PANEL_AXES_RECT["left"]) / (PANEL_AXES_RECT["right"] - PANEL_AXES_RECT["left"])


def row_figsize(n):
    """Canvas for a 1xn row: n panels wide, one panel tall."""
    return (n * PANEL_W, PANEL_H)


def row_axes_rect(n):
    return {
        "left": PANEL_AXES_RECT["left"] / n, "right": 1 - PANEL_AXES_RECT["left"] / n,
        "top": PANEL_AXES_RECT["top"], "bottom": PANEL_AXES_RECT["bottom"],
        "wspace": _WSPACE,
    }


def save_row(fig, path, n=None, dpi=600):
    """save_panel for a 1xn row. n defaults to the figure's subplot count."""
    if n is None:
        n = len(fig.axes)
    fig.subplots_adjust(**row_axes_rect(n))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg", dpi=dpi)
    print(f"[ok] saved: {path}")


# n=2 aliases, kept because make_hidden_subspace_fig.py already imports them.
WIDE_FIGSIZE = row_figsize(2)
WIDE_AXES_RECT = row_axes_rect(2)


def save_wide(fig, path, dpi=600):
    return save_row(fig, path, n=2, dpi=dpi)


# ---------------------------------------------------------------------------
# Training-curve panels (the 4 on-policy + 4 off-policy score curves).
#
# Those eight scripts carry byte-identical plotting blocks, so the style lives
# here and each script only supplies its data, y limits and teacher line.
# Line widths are scaled for the smaller canvas: 2.6 pt on the old 7x5.5 figure
# is proportionally much heavier on a 4.9x3.03 panel, where three curves plus a
# teacher line have to stay separable.
# ---------------------------------------------------------------------------
CURVE_RAW_LW = 1.0
CURVE_RAW_ALPHA = 0.28
CURVE_SMOOTH_LW = 2.4
CURVE_TEACHER_LW = 1.5

# 2 pt below the sublayer panels: these eight figures are the same plot repeated
# across benchmarks, so the title and axis labels are read once and then only
# carry the "which one is this" information -- they should not compete with the
# curves. The titles themselves are kept terse ("<benchmark> · <model>", the
# on/off-policy distinction lives in the caption); the widest is 187 pt on a
# 353 pt canvas, so there is room to lengthen one if a benchmark needs it.
FS_CURVE_TITLE = FS_TITLE - 2
FS_CURVE_AXLABEL = FS_AXLABEL - 2


def curve_panel_rc():
    """panel_rc() plus the darker axes/grid these curve figures already used."""
    panel_rc()
    plt.rcParams.update({
        "axes.edgecolor": "#444444", "axes.linewidth": 1.1,
        "grid.color": "#dddddd", "grid.linewidth": 0.8,
    })


def style_curve_axes(ax, title, xlabel="Training step", ylabel="Accuracy"):
    """Labels, title and spines at the shared curve-panel font sizes."""
    ax.set_xlabel(xlabel, fontsize=FS_CURVE_AXLABEL, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FS_CURVE_AXLABEL, labelpad=6)
    ax.set_title(title, fontsize=FS_CURVE_TITLE, pad=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def curve_legend(ax, loc="lower right"):
    leg = ax.legend(loc=loc, frameon=True, fancybox=True, framealpha=0.92,
                    edgecolor="#cccccc", fontsize=FS_LEGEND, handlelength=1.5,
                    handletextpad=0.4, labelspacing=0.25, borderpad=0.4,
                    columnspacing=0.9)
    leg.get_frame().set_linewidth(0.8)
    return leg
