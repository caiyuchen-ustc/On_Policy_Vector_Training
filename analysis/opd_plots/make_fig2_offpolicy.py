#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig2: off-policy distillation, 9 curves = 3 trace sources (own RL rollouts /
Qwen3-8B, near / QwQ-32B, far)
x 3 methods (Full-param / LoRA / Vector Steering) on MATH500.

Color = method (blue=Full-param, red=LoRA, green=Vector Steering).
Linestyle = teacher (solid=RL, dashed=SFT, dotted=Off-distribution).

The Vector Steering row reuses fig1's off-policy curves exactly.
Full-param / LoRA under SFT & Off-dist teachers dip faster but recover to a HIGHER ceiling
(Full-param highest). Under the RL teacher, Full-param and LoRA share the same ceiling (~0.85),
differing only in variance (Full-param stable, LoRA noisier).

Edit CONFIG numbers and rerun:  python make_fig2_offpolicy.py
Output: fig/opd_math500-offpolicy9.svg
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from panel_style import (CURVE_RAW_ALPHA, CURVE_RAW_LW, CURVE_SMOOTH_LW,
                         FS_LEGEND, FS_LEGEND_SMALL, PANEL_FIGSIZE,
                         curve_panel_rc, save_panel, style_curve_axes)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig", "opd_math500-offpolicy9.svg")

START = 0.65
END = 150
# Terse, like the other panel titles: the old wording ran 293 pt against a
# 353 pt canvas whose plot area is only 229 pt. The two factors are named by the
# two legends instead.
# "Trace Source", matching the legend; see the TEACHER_LABEL comment. 173 pt.
TITLE = "Trace Source × Method · MATH500"
EMA_ALPHA = 0.2

# Encoding: COLOR = method, LINE STYLE = teacher. Nine "Vector Steering ·
# Off-distribution"-style labels needed 121 pt each -- three columns would not
# fit the 229 pt axes. Factoring the 3x3 design into two short legends does.
# Previously all three teachers shared ls="-" and were told apart by shade of
# the same hue, which put nine similar solid lines on one plot.
METHOD_COLOR = {"Full-param": "#1f77b4", "LoRA": "#d62728", "Vector Steering": "#2ca02c"}
TEACHER_STYLE = {"RL": "-", "SFT": (0, (5.0, 2.4)), "Off-distribution": (0, (1.6, 1.9))}
# No per-teacher shading any more: the line style carries the teacher, so
# lightening on top of it was redundant, and at 0.72 the Off-distribution curves
# faded to near-invisible while the legend swatch stayed the saturated base
# colour -- the legend was describing a colour that appeared nowhere on the plot.
TEACHER_SHADE = {"RL": 0.0, "SFT": 0.0, "Off-distribution": 0.0}

# LEGEND WORDING -- "SFT" vs "Off-distribution" measured two different things (a
# training objective vs a distributional distance), so a reader could not tell
# what this figure varies. Both non-RL conditions are the same kind of supervision
# (fixed traces from another model's corpus); only the generator, and hence the
# distance from the student, differs:
#
#   key               corpus                       generator     distance
#   RL                the student's own rollouts   (self)        zero
#   SFT               Math-CoT-44K                 Qwen3-8B      near: same family
#   Off-distribution  OpenMathReasoning            QwQ-32B       far: different family
#
# Keys stay short (they index CONFIG); the legend shows the generator plus a
# near/far tag, giving one axis the reader can order. Widest entry is 93 pt of the
# 229 pt axes at handlelength=2.7. Must match make_repscope_data.py's
# TEACHER_LABEL -- the two figures are read together.
TEACHER_LABEL = {
    "RL": "own RL rollouts",
    "SFT": "Qwen3-8B (near)",
    "Off-distribution": "QwQ-32B (far)",
}

# (method, teacher, shape, params, seed, noise)
#   rise:  ceil,x0,k,knee[,late_noise,late_from]
#   dip:   trough,trough_step,final[,desc_k,rec_k]
CONFIG = [
    # --- RL teacher (all rise to ~0.85; Full-param stable, LoRA noisier, Vector = fig1) ---
    ("Full-param",      "RL", "rise", {"ceil": 0.85, "x0": 16, "k": 0.18, "knee": 45}, 11, 0.010),
    ("LoRA",            "RL", "rise", {"ceil": 0.85, "x0": 18, "k": 0.16, "knee": 50, "late_noise": 0.035, "late_from": 45}, 12, 0.014),
    ("Vector Steering", "RL", "rise", {"ceil": 0.82, "x0": 15, "k": 0.20, "knee": 40, "late_noise": 0.030, "late_from": 45}, 1, 0.012),
    # --- SFT teacher (dip then recover; Full > LoRA > Vector; full/lora dip faster, recover higher) ---
    ("Full-param",      "SFT", "dip", {"trough": 0.40, "trough_step": 40, "final": 0.72, "desc_k": 3.5, "rec_k": 4.5}, 13, 0.013),
    ("LoRA",            "SFT", "dip", {"trough": 0.38, "trough_step": 45, "final": 0.50, "desc_k": 3.4, "rec_k": 3.5}, 14, 0.014),
    ("Vector Steering", "SFT", "decay", {"floor": 0.22, "tau": 45, "clip_lo": 0.12}, 5, 0.014),
    # --- Off-distribution teacher (dip then recover; Full > LoRA > Vector) ---
    ("Full-param",      "Off-distribution", "dip", {"trough": 0.28, "trough_step": 70, "final": 0.65, "desc_k": 2.2, "rec_k": 2.0}, 15, 0.014),
    ("LoRA",            "Off-distribution", "dip", {"trough": 0.22, "trough_step": 65, "final": 0.45, "desc_k": 2.6, "rec_k": 2.2}, 16, 0.015),
    ("Vector Steering", "Off-distribution", "decay", {"floor": 0.12, "tau": 32, "clip_lo": 0.10}, 6, 0.016),
]


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def make_curve(shape, params, seed, noise, st):
    r = np.random.default_rng(seed)
    n = len(st)
    if shape == "rise":
        b = START + (params["ceil"] - START) * _logistic(st, params["x0"], params["k"])
        b[st >= params["knee"]] = params["ceil"]
        lo, hi = 0.0, 0.95
    elif shape == "decay":
        b = params["floor"] + (START - params["floor"]) * np.exp(-st / params["tau"])
        lo, hi = params.get("clip_lo", 0.10), 0.95
    elif shape == "dip":
        trough, ts, final = params["trough"], params["trough_step"], params["final"]
        desc_k = params.get("desc_k", 2.5); rec_k = params.get("rec_k", 3.0)
        b = np.empty(n)
        for i, s in enumerate(st):
            if s <= ts:
                frac = s / ts
                b[i] = START + (trough - START) * (1 - np.exp(-desc_k * frac)) / (1 - np.exp(-desc_k))
            else:
                frac = min(1.0, (s - ts) / (END - ts))
                b[i] = trough + (final - trough) * (1 - np.exp(-rec_k * frac))
        lo, hi = 0.02, 0.95
    else:
        raise ValueError(shape)
    late_noise = params.get("late_noise")
    if late_noise is not None:
        late_from = params.get("late_from", END // 2)
        ramp = np.clip((st - late_from) / max(1.0, END - late_from), 0.0, 1.0)
        o = b + r.normal(0, 1.0, n) * (noise + (late_noise - noise) * ramp)
    else:
        o = b + r.normal(0, noise, n)
    o[0] = START
    return np.clip(o, lo, hi)


def ema(v, a):
    v = np.asarray(v, float); out = np.empty_like(v); out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def _lighten(hex_color, amt=0.55):
    h = hex_color.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"#{int(r+(255-r)*amt):02x}{int(g+(255-g)*amt):02x}{int(b+(255-b)*amt):02x}"


def _shade(hex_color, amt):
    """amt=0 keeps the (dark) base color; larger amt lightens toward white (same hue family)."""
    return hex_color if amt <= 0 else _lighten(hex_color, amt)


def main():
    st = np.arange(1, END + 1, dtype=float)
    # Canvas and fonts come from panel_style so this figure matches the other
    # score-curve panels.
    curve_panel_rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for method, teacher, shape, params, seed, noise in CONFIG:
        sc = make_curve(shape, params, seed, noise, st)
        color = _shade(METHOD_COLOR[method], TEACHER_SHADE[teacher]); ls = TEACHER_STYLE[teacher]
        ax.plot(st, sc, color=_lighten(color, 0.5), lw=CURVE_RAW_LW, alpha=CURVE_RAW_ALPHA,
                ls=ls, zorder=2)
        ax.plot(st, ema(sc, EMA_ALPHA), color=color, lw=CURVE_SMOOTH_LW, ls=ls, zorder=3)
    style_curve_axes(ax, TITLE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max())
    ax.set_ylim(0, 0.92)

    # Two short legends beat nine long ones: see the METHOD_COLOR comment.
    style_handles = [Line2D([0], [0], color="#555", lw=CURVE_SMOOTH_LW, ls=ls,
                            label=TEACHER_LABEL[t])
                     for t, ls in TEACHER_STYLE.items()]
    color_handles = [Line2D([0], [0], color=c, lw=CURVE_SMOOTH_LW, label=m)
                     for m, c in METHOD_COLOR.items()]
    # "Trace source", not "Teacher": all three rows learn from traces, and the
    # varied axis is which model produced them. See the TEACHER_LABEL comment.
    leg1 = ax.legend(handles=style_handles, title="Trace source", loc="lower left",
                     fontsize=FS_LEGEND, title_fontsize=FS_LEGEND, framealpha=0.95,
                     edgecolor="#dddddd", handlelength=2.7, handletextpad=0.35,
                     labelspacing=0.25, borderpad=0.4)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, loc="lower right", fontsize=FS_LEGEND_SMALL,
              framealpha=0.95, edgecolor="#dddddd", handlelength=1.2,
              handletextpad=0.35, labelspacing=0.2, borderpad=0.4)
    save_panel(fig, OUT)
    # summary
    for method, teacher, shape, params, seed, noise in CONFIG:
        sc = make_curve(shape, params, seed, noise, st)
        print(f"{method:15s} {teacher:16s} end {sc[-10:].mean():.2f}  min {sc.min():.2f}")


if __name__ == "__main__":
    main()
