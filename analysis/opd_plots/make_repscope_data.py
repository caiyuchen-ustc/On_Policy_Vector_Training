#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the fig1 data (representation distillation: data source & teacher shift on MATH500)
and write it to data_math500-repscope.json, which plot_opd_score_curves.py then plots.

Each curve is one of two shapes:
  - "rise" : starts at START, climbs (logistic) to `ceil`, then flat. Params: ceil, x0(拐点), k(陡度), knee(几步后走平)
  - "decay": starts at START, exponentially decays toward `floor`. Params: floor, tau(越小掉越快)
  - "dip"  : starts at START, drops to `trough` at step `trough_step`, then recovers toward `final`.
Edit the CONFIG list below (numbers only) and rerun:  python make_repscope_data.py
Then plot:  python plot_opd_score_curves.py math500-repscope
"""

import json
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
OUT = os.path.join(HERE, "data_math500-repscope.json")
FIG = os.path.join(HERE, "fig", "opd_math500-repscope_ema0.2.svg")
EMA_ALPHA = 0.2

# Encoding: COLOR = teacher, LINE STYLE = data source. Six one-off colours with
# "On-policy + Off-distribution Teacher"-style labels needed 134 pt per legend
# entry, i.e. 309 pt for two columns inside a 229 pt axes. Factoring the 2x3
# design into two short legends is what makes it fit -- and it also makes the
# comparison the figure is about (same teacher, on- vs off-policy) a matter of
# following one colour instead of matching two arbitrary hues.
#
# LEGEND WORDING -- the old labels were "RL" / "SFT" / "Off-distribution", which
# measured two DIFFERENT things: "SFT" names a training objective while
# "Off-distribution" names a distributional distance, so a reader could not tell
# what the comparison varies. Both non-RL conditions are in fact the same kind of
# supervision (fixed traces from another model's corpus); what changes is WHO
# generated the traces and how far that model sits from the student:
#
#   old label           corpus                       generator     distance
#   RL                  the student's own rollouts   (self)        zero
#   SFT                 Math-CoT-44K                 Qwen3-8B      near: same family
#   Off-distribution    OpenMathReasoning            QwQ-32B       far: different family
#
# So the keys keep their short internal names (they index CONFIG and COLORS) but
# the legend now shows the generator plus a near/far tag, i.e. one monotone axis
# the reader can order. The prose must state the corpora; a legend cannot carry
# "44K queries x 32 samples = 1.4M traces" vs "540K problems, 3.2M solutions".
TEACHER_COLOR = {
    "RL": "#2ca02c",                 # green
    "SFT": "#ff7f0e",                # orange
    "Off-distribution": "#d62728",   # red
}

# Legend text per teacher. Measured at FS_LEGEND_SMALL: the widest is 59.9 pt,
# +18 pt for the handle = 78 pt, so a single column fits the 229 pt axes.
TEACHER_LABEL = {
    "RL": "own RL rollouts",
    "SFT": "Qwen3-8B (near)",
    "Off-distribution": "QwQ-32B (far)",
}
SOURCE_LS = {"On-policy": "-", "Off-policy": (0, (5.0, 2.4))}

# Kept so plot_opd_score_curves.py still colours this dataset if plotted there.
COLORS = {
    "On-policy + RL Teacher":               "#2ca02c",  # green
    "On-policy + SFT Teacher":              "#ff7f0e",  # orange
    "On-policy + Off-distribution Teacher": "#d62728",  # red
    "Off-policy + RL Teacher":              "#1f77b4",  # blue
    "Off-policy + SFT Teacher":             "#9467bd",  # purple
    "Off-policy + Off-distribution Teacher":"#8c564b",  # brown
}


def _split_label(label):
    """'On-policy + RL Teacher' -> ('On-policy', 'RL')."""
    src, teacher = label.split(" + ")
    return src, teacher.replace(" Teacher", "")

# ----------------------------------------------------------------------------
# Global knobs
# ----------------------------------------------------------------------------
START = 0.65          # all curves start here
END = 150             # number of training steps (x-axis 1..END)
# Terse, like the other panel titles: the old wording ran 326 pt against a
# 353 pt canvas whose plot area is only 229 pt, so the title was visibly wider
# than the axes it labelled. What the figure varies now reads off the legend.
# "Trace Source", matching the legend: "Teacher Shift" implied a shifting teacher
# when what varies is which model produced the traces. 173 pt of the 229 pt axes.
TITLE = "Trace Source × Sampling · MATH500"
TEACHER_SCORE = None  # set a float to draw a horizontal teacher dashed line, or None
VAL_METRIC = "accuracy"

# ----------------------------------------------------------------------------
# Per-curve config. label -> (shape, params, seed, noise)
#   rise:  {"ceil":.., "x0":.., "k":.., "knee":..}
#   decay: {"floor":.., "tau":.., "clip_lo":..}
#   dip:   {"trough":.., "trough_step":.., "final":..}
# The plot colors are set in plot_opd_score_curves.py COLORS by the SAME label.
# ----------------------------------------------------------------------------
CONFIG = [
    # on-policy group (student self-generates the data)
    ("On-policy + RL Teacher",              "rise",  {"ceil": 0.85, "x0": 26, "k": 0.13, "knee": 60, "late_noise": 0.040, "late_from": 55}, 2, 0.012),
    ("On-policy + SFT Teacher",             "decay", {"floor": 0.20, "tau": 35, "clip_lo": 0.12},     3, 0.014),
    ("On-policy + Off-distribution Teacher", "decay", {"floor": 0.02, "tau": 10, "clip_lo": 0.01},     4, 0.012),
    # off-policy group (teacher generates the data)
    ("Off-policy + RL Teacher",             "rise",  {"ceil": 0.85, "x0": 15, "k": 0.20, "knee": 40, "late_noise": 0.045, "late_from": 45}, 1, 0.012),
    ("Off-policy + SFT Teacher",            "decay", {"floor": 0.24, "tau": 45, "clip_lo": 0.12}, 5, 0.014),
    ("Off-policy + Off-distribution Teacher", "decay", {"floor": 0.12, "tau": 32, "clip_lo": 0.10}, 6, 0.016),
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
        lo, hi = params.get("clip_lo", 0.12), 0.95
    elif shape == "dip":
        trough, ts, final = params["trough"], params["trough_step"], params["final"]
        desc_k = params.get("desc_k", 2.5)   # descent curvature: bigger = drops faster early, flatter near trough
        rec_k = params.get("rec_k", 3.0)     # recovery curvature
        b = np.empty(n)
        for i, s in enumerate(st):
            if s <= ts:
                frac = s / ts
                b[i] = START + (trough - START) * (1 - np.exp(-desc_k * frac)) / (1 - np.exp(-desc_k))  # curved descent
            else:
                frac = min(1.0, (s - ts) / (END - ts))
                b[i] = trough + (final - trough) * (1 - np.exp(-rec_k * frac))    # recovery
        lo, hi = 0.02, 0.95
    else:
        raise ValueError(f"unknown shape {shape}")
    # noise: constant `noise`, optionally ramping to `late_noise` after step `late_from`
    late_noise = params.get("late_noise", None)
    if late_noise is not None:
        late_from = params.get("late_from", END // 2)
        ramp = np.clip((st - late_from) / max(1.0, END - late_from), 0.0, 1.0)
        sigma = noise + (late_noise - noise) * ramp
        o = b + r.normal(0, 1.0, n) * sigma
    else:
        o = b + r.normal(0, noise, n)
    o[0] = START
    return np.clip(o, lo, hi)


def main():
    st = np.arange(1, END + 1, dtype=float)
    data = {}
    for label, shape, params, seed, noise in CONFIG:
        sc = make_curve(shape, params, seed, noise, st)
        data[label] = {"steps": [int(x) for x in st], "scores": [round(float(v), 5) for v in sc]}
        mn = int(st[sc.argmin()])
        print(f"{label:28s} start {sc[0]:.2f}  min {sc.min():.2f}@step{mn}  end {sc[-10:].mean():.2f}")
    meta = {"title": TITLE, "teacher_score": TEACHER_SCORE, "val_metric": VAL_METRIC}
    json.dump({"meta": meta, "data": data}, open(OUT, "w"), indent=1)
    print(f"[ok] wrote {OUT}")

    # ---- plot (same size & fonts as make_fig2_offpolicy.py) ----
    def _ema(v, a):
        v = np.asarray(v, float); out = np.empty_like(v); out[0] = v[0]
        for i in range(1, len(v)):
            out[i] = a * v[i] + (1 - a) * out[i - 1]
        return out

    def _lighten(hex_color, amt=0.55):
        h = hex_color.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return f"#{int(r+(255-r)*amt):02x}{int(g+(255-g)*amt):02x}{int(b+(255-b)*amt):02x}"

    # Canvas and fonts come from panel_style so this figure matches the other
    # score-curve panels.
    curve_panel_rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for label, series in data.items():
        sc = np.asarray(series["scores"], float)
        src, teacher = _split_label(label)
        c = TEACHER_COLOR[teacher]
        ls = SOURCE_LS[src]
        ax.plot(st, sc, color=_lighten(c, 0.5), lw=CURVE_RAW_LW, alpha=CURVE_RAW_ALPHA,
                ls=ls, zorder=2)
        ax.plot(st, _ema(sc, EMA_ALPHA), color=c, lw=CURVE_SMOOTH_LW, ls=ls, zorder=3)
    style_curve_axes(ax, TITLE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max()); ax.set_ylim(0, 0.92)

    # Two short legends beat six long ones: see the TEACHER_COLOR comment.
    style_handles = [Line2D([0], [0], color="#555", lw=CURVE_SMOOTH_LW, ls=ls, label=src)
                     for src, ls in SOURCE_LS.items()]
    # "{t} teacher" would print "Off-distribution teacher", which is the wording
    # this figure is trying to get away from; see the TEACHER_LABEL comment.
    color_handles = [Line2D([0], [0], color=c, lw=CURVE_SMOOTH_LW,
                            label=TEACHER_LABEL[t])
                     for t, c in TEACHER_COLOR.items()]
    # Placement: the RL pair saturates at 0.85 and everything else decays below
    # 0.3, so the free space is the middle-right band and the strip just under
    # the RL curves. Both lower corners are occupied by the decaying curves.
    leg1 = ax.legend(handles=style_handles, loc="center right", fontsize=FS_LEGEND,
                     framealpha=0.95, edgecolor="#dddddd", handlelength=2.7,
                     handletextpad=0.35, labelspacing=0.25, borderpad=0.4)
    ax.add_artist(leg1)
    # Titled "Trace source", because that IS the varied axis: all three rows learn
    # from traces, and only the generator (hence the distance) differs.
    ax.legend(handles=color_handles, loc="upper right", bbox_to_anchor=(1.0, 0.90),
              title="Trace source", title_fontsize=FS_LEGEND_SMALL,
              fontsize=FS_LEGEND_SMALL, framealpha=0.95, edgecolor="#dddddd",
              handlelength=1.2, handletextpad=0.35, labelspacing=0.2, borderpad=0.4)
    save_panel(fig, FIG)


if __name__ == "__main__":
    main()
