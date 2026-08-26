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
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_math500-repscope.json")
FIG = os.path.join(HERE, "fig", "opd_math500-repscope_ema0.2.svg")
EMA_ALPHA = 0.2

# same figure style as make_fig2_offpolicy.py (identical size & fonts)
COLORS = {
    "On-policy + RL Teacher":               "#2ca02c",  # green
    "On-policy + SFT Teacher":              "#ff7f0e",  # orange
    "On-policy + Off-distribution Teacher": "#d62728",  # red
    "Off-policy + RL Teacher":              "#1f77b4",  # blue
    "Off-policy + SFT Teacher":             "#9467bd",  # purple
    "Off-policy + Off-distribution Teacher":"#8c564b",  # brown
}

# ----------------------------------------------------------------------------
# Global knobs
# ----------------------------------------------------------------------------
START = 0.65          # all curves start here
END = 150             # number of training steps (x-axis 1..END)
TITLE = "Performance of Different Data Sources & Teacher Shifts on MATH500"
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

    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans", "axes.grid": True,
                         "grid.color": "#dddddd", "grid.linewidth": 0.8,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "figure.dpi": 600, "savefig.dpi": 600})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, series in data.items():
        sc = np.asarray(series["scores"], float)
        c = COLORS.get(label, "#333333")
        ax.plot(st, sc, color=_lighten(c, 0.5), lw=1.0, alpha=0.22, zorder=2)
        ax.plot(st, _ema(sc, EMA_ALPHA), color=c, lw=2.3, zorder=3, label=label)
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=17, labelpad=8)
    ax.set_title(TITLE, fontsize=16, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max()); ax.set_ylim(0, 0.92)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9, edgecolor="#cccccc",
              handlelength=1.6, handletextpad=0.4, labelspacing=0.3, columnspacing=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
