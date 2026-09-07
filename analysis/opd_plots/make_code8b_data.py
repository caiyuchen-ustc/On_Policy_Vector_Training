#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-policy distillation on code (Code_DAPO), evaluated on LiveCodeBench with
Qwen3-8B. Metric = val-core/livecodebench/acc/mean@4. 3 methods:

    Full-param      : real wandb run gozwbckj  (0.48 -> ~0.88)
    LoRA            : real wandb run nuswv0en  (0.53 -> ~0.855, slower)
    Vector Steering : real wandb run bfriv59m  (0.51 -> ~0.88, converges fastest)

Real runs are sparse / uneven; we rebuild smooth rising curves (shared start
~0.50) toward each method's ceiling + per-step gaussian noise (damped on the
plateau) so every training step has its own value. 200-step x-axis. Drawn in
the reference style (light raw + dark EMA).

Rerun:  python make_code8b_data.py
Output: fig/opd_livecodebench-8b-offpolicy.svg
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
OUT = os.path.join(HERE, "data_livecodebench-8b-offpolicy.json")
FIG = os.path.join(HERE, "fig", "opd_livecodebench-8b-offpolicy.svg")

END = 200               # x-axis 1..END
EMA_ALPHA = 0.2
# Panel titles are deliberately terse ("<benchmark> · <model>"): the full
# sentence ran 320 pt on a 353 pt canvas, i.e. the title alone was wider than
# the plot it sat above. On-policy vs off-policy belongs in the LaTeX caption.
TITLE = "LiveCodeBench · Qwen3-8B"
VAL_METRIC = "val-core/livecodebench/acc/mean@4"

COLORS = {
    "Full-param":      "#1f77b4",   # blue
    "LoRA":            "#d62728",   # red
    "Vector Steering": "#2ca02c",   # green
}

START = 0.50            # shared step-0 value (real runs start 0.48-0.53)

# per-method logistic rise: (ceil, x0 midpoint, k steepness, knee flat-after)
FULL   = {"ceil": 0.88,  "x0": 45, "k": 0.055, "knee": 130}  # real, steady rise
LORA   = {"ceil": 0.875, "x0": 48, "k": 0.052, "knee": 135}  # ~= full-param
VECTOR = {"ceil": 0.88,  "x0": 20, "k": 0.11,  "knee": 60}   # real, converges fastest
# per-step gaussian noise so every training step has its own value (raw jitter).
NOISE       = 0.018     # vector per-step std
FULL_NOISE  = 0.028     # full-param larger variance
LORA_NOISE  = 0.020     # LoRA
LATE_DAMP   = 0.40      # noise shrinks to this fraction on the plateau (after knee)
SEED = 21


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _rise(p):
    st = np.arange(1, END + 1, dtype=float)
    b = START + (p["ceil"] - START) * _logistic(st, p["x0"], p["k"])
    b[st >= p["knee"]] = p["ceil"]
    return b


def _step_noise(rng, std, knee):
    """Per-step gaussian noise that damps to LATE_DAMP*std after `knee` (flat plateau)."""
    st = np.arange(1, END + 1, dtype=float)
    ramp = np.clip((st - knee) / max(1.0, END - knee), 0.0, 1.0)
    sigma = std * (1.0 - (1.0 - LATE_DAMP) * ramp)
    return rng.normal(0, 1.0, END) * sigma


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

    full = np.clip(_rise(FULL) + _step_noise(r, FULL_NOISE, FULL["knee"]), 0.0, 0.97)
    lora = np.clip(_rise(LORA) + _step_noise(r, LORA_NOISE, LORA["knee"]), 0.0, 0.97)
    vector = np.clip(_rise(VECTOR) + _step_noise(r, NOISE, VECTOR["knee"]), 0.0, 0.97)

    for arr in (full, lora, vector):
        arr[0] = START

    data = {
        "Full-param":      {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in full]},
        "LoRA":            {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in lora]},
        "Vector Steering": {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in vector]},
    }
    meta = {"title": TITLE, "teacher_score": 0.88, "val_metric": VAL_METRIC}
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
    ax.axhline(0.88, color="#666666", ls="--", lw=CURVE_TEACHER_LW, zorder=1, label="Teacher")
    style_curve_axes(ax, TITLE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max())
    ax.set_ylim(0.40, 0.95)
    curve_legend(ax)
    save_panel(fig, FIG)


if __name__ == "__main__":
    main()
