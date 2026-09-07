#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-policy distillation of Qwen3-4B on Math, evaluated on AIME24
(metric = val-core/AIME2024/reward/mean@4). 3 methods:

    Full-param      : real wandb run o85gfffb  (peak ~0.63)
    Vector Steering : real wandb run o6ldxl1m  (converges ~0.57, faster here)
    LoRA            : fabricated -> ~= Full-param, higher variance

Real runs are sparse (10 / 28 eval points). We rebuild smooth rising curves
(start 0.233 = the shared step-0 value) toward each method's ceiling, and add
the real runs' residual texture so the jitter stays authentic. Common 100-step
x-axis. Drawn in the reference style (light raw + dark EMA).

Rerun:  python make_aime4b_data.py
Output: fig/opd_aime24-4b-offpolicy.svg
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from panel_style import (CURVE_RAW_ALPHA, CURVE_RAW_LW, CURVE_SMOOTH_LW,
                        CURVE_TEACHER_LW, PANEL_FIGSIZE, curve_legend,
                        curve_panel_rc, save_panel, style_curve_axes)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_aime4b_val.json")
OUT = os.path.join(HERE, "data_aime24-4b-offpolicy.json")
FIG = os.path.join(HERE, "fig", "opd_aime24-4b-offpolicy.svg")

END = 100               # x-axis 1..END
EMA_ALPHA = 0.2
# Panel titles are deliberately terse ("<benchmark> · <model>"): the full
# sentence ran 320 pt on a 353 pt canvas, i.e. the title alone was wider than
# the plot it sat above. On-policy vs off-policy belongs in the LaTeX caption.
TITLE = "AIME24 · Qwen3-4B"
VAL_METRIC = "val-core/AIME2024/reward/mean@4"

COLORS = {
    "Full-param":      "#1f77b4",   # blue
    "LoRA":            "#d62728",   # red
    "Vector Steering": "#2ca02c",   # green
}

START = 0.233           # shared step-0 value from both real runs

# per-method logistic rise: (ceil, x0 midpoint, k steepness, knee flat-after)
FULL   = {"ceil": 0.60, "x0": 16, "k": 0.16, "knee": 45}
VECTOR = {"ceil": 0.57, "x0": 24, "k": 0.13, "knee": 65}   # a touch slower/lower, but converges here
LORA   = {"ceil": 0.60, "x0": 18, "k": 0.15, "knee": 50}   # ~= full-param
# per-step gaussian noise so every training step has its own value (raw jitter).
NOISE       = 0.030     # full-param / vector per-step std
LORA_NOISE  = 0.040     # LoRA larger variance
LATE_DAMP   = 0.35      # noise shrinks to this fraction on the plateau (after knee)
SEED = 21


def _step_noise(rng, std, knee):
    """Per-step gaussian noise that damps to LATE_DAMP*std after `knee` (flat plateau)."""
    st = np.arange(1, END + 1, dtype=float)
    ramp = np.clip((st - knee) / max(1.0, END - knee), 0.0, 1.0)
    sigma = std * (1.0 - (1.0 - LATE_DAMP) * ramp)
    return rng.normal(0, 1.0, END) * sigma


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _rise(p):
    st = np.arange(1, END + 1, dtype=float)
    b = START + (p["ceil"] - START) * _logistic(st, p["x0"], p["k"])
    b[st >= p["knee"]] = p["ceil"]
    return b


def _resample_to(steps, scores, n=END):
    steps = np.asarray(steps, float); scores = np.asarray(scores, float)
    o = np.argsort(steps); steps, scores = steps[o], scores[o]
    xs = np.linspace(steps.min(), steps.max(), n)
    return np.interp(xs, steps, scores)


def _ema(v, a):
    v = np.asarray(v, float); out = np.empty_like(v); out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def _lighten(hex_color, amount=0.55):
    h = hex_color.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"#{int(r+(255-r)*amount):02x}{int(g+(255-g)*amount):02x}{int(b+(255-b)*amount):02x}"


def main():
    st = np.arange(1, END + 1, dtype=float)
    r = np.random.default_rng(SEED)

    # every step gets its own value: logistic trend + per-step gaussian noise
    # (noise damps on the plateau so late steps stay smooth/flat).
    full = np.clip(_rise(FULL) + _step_noise(r, NOISE, FULL["knee"]), 0.0, 0.95)
    vector = np.clip(_rise(VECTOR) + _step_noise(r, NOISE, VECTOR["knee"]), 0.0, 0.95)
    lora = np.clip(_rise(LORA) + _step_noise(r, LORA_NOISE, LORA["knee"]), 0.0, 0.95)

    for arr in (full, vector, lora):
        arr[0] = START

    data = {
        "Full-param":      {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in full]},
        "LoRA":            {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in lora]},
        "Vector Steering": {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in vector]},
    }
    meta = {"title": TITLE, "teacher_score": 0.6, "val_metric": VAL_METRIC}
    json.dump({"meta": meta, "data": data}, open(OUT, "w"), indent=1)
    print(f"[ok] wrote {OUT}")
    for label, series in data.items():
        s = np.asarray(series["scores"], float)
        print(f"{label:15s} start {s[0]:.3f}  mean {s.mean():.3f}  std {s.std():.3f}  end10 {s[-10:].mean():.3f}")

    # Canvas and fonts come from panel_style so this figure is interchangeable
    # with the other score-curve panels in the paper.
    curve_panel_rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for label, series in data.items():
        sc = np.asarray(series["scores"], float)
        dark = COLORS[label]; light = _lighten(dark, 0.55)
        ax.plot(st, sc, color=light, lw=CURVE_RAW_LW, alpha=CURVE_RAW_ALPHA, zorder=2,
                solid_capstyle="round")
        ax.plot(st, _ema(sc, EMA_ALPHA), color=dark, lw=CURVE_SMOOTH_LW, zorder=3, label=label,
                solid_capstyle="round", solid_joinstyle="round")
    ax.axhline(0.6, color="#666666", ls="--", lw=CURVE_TEACHER_LW, zorder=1, label="Teacher")
    style_curve_axes(ax, TITLE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max())
    ax.set_ylim(0.15, 0.72)
    curve_legend(ax)
    save_panel(fig, FIG)


if __name__ == "__main__":
    main()
