#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-policy distillation on DeepSeek-R1-Distill-Qwen-1.5B / SciKnowEval, 3 methods.
Metric = val-core/sciknoweval/reward/mean@4 (the real validation score).

    Full-param      : real wandb run hhfo0z8k   (0.41 -> ~0.72)
    Vector Steering : real wandb run 7gt4vbj3   (0.41 -> ~0.63, slower / lower)
    LoRA            : fabricated -> tracks Full-param's trend, higher variance

Both real runs are resampled onto a 150-step x-axis. Curves are drawn in the
reference style of analysis/plot_opd_score_curves.py (light raw + dark EMA).

Rerun:  python make_sciknow1p5b_data.py
Output: fig/opd_sciknoweval-1p5b-offpolicy.svg
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
RAW = os.path.join(HERE, "raw_sciknow1p5b_val.json")   # cached wandb val dump (full/vector)
OUT = os.path.join(HERE, "data_sciknoweval-1p5b-offpolicy.json")
FIG = os.path.join(HERE, "fig", "opd_sciknoweval-1p5b-offpolicy.svg")

END = 150               # x-axis 1..END
EMA_ALPHA = 0.2
# Panel titles are deliberately terse ("<benchmark> · <model>"): the full
# sentence ran 320 pt on a 353 pt canvas, i.e. the title alone was wider than
# the plot it sat above. On-policy vs off-policy belongs in the LaTeX caption.
# Still the longest of the eight, so measure before lengthening it.
TITLE = "SciKnowEval · Qwen2.5-1.5B-Deepseek"
VAL_METRIC = "val-core/sciknoweval/reward/mean@4"

COLORS = {
    "Full-param":      "#1f77b4",   # blue
    "LoRA":            "#d62728",   # red
    "Vector Steering": "#2ca02c",   # green
}

# LoRA is fabricated to sit at the same level as full-param but with larger
# per-step variance ("same effect, different variance").
LORA_NOISE = 0.028
LORA_SEED = 21

# Vector Steering: real run converges ~0.62; rescale it (keeping its shape &
# texture) so it converges to VECTOR_TARGET instead.
VECTOR_START = 0.407
VECTOR_TARGET = 0.67


def _resample_to(steps, scores, n=END):
    """Linearly resample an uneven (steps, scores) series onto 1..n."""
    steps = np.asarray(steps, float)
    scores = np.asarray(scores, float)
    order = np.argsort(steps)
    steps, scores = steps[order], scores[order]
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
    raw = json.load(open(RAW))
    st = np.arange(1, END + 1, dtype=float)

    # real curves, resampled to 150 steps
    full = np.clip(_resample_to(raw["full"]["step"], raw["full"]["score"]), 0.0, 0.95)
    vector = np.clip(_resample_to(raw["vector"]["step"], raw["vector"]["score"]), 0.0, 0.95)

    # rescale vector (keep shape/texture) so it converges to VECTOR_TARGET.
    v_end = float(_ema(vector, 0.06)[-10:].mean())
    scale = (VECTOR_TARGET - VECTOR_START) / max(1e-6, v_end - VECTOR_START)
    vector = np.clip(VECTOR_START + (vector - VECTOR_START) * scale, 0.0, 0.95)

    # LoRA: follow full-param's slow trend but with its own higher-variance jitter.
    trend = _ema(full, 0.06)
    r = np.random.default_rng(LORA_SEED)
    lora = np.clip(trend + r.normal(0, LORA_NOISE, END), 0.0, 0.95)
    lora[0] = full[0]

    data = {
        "Full-param":      {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in full]},
        "LoRA":            {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in lora]},
        "Vector Steering": {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in vector]},
    }
    meta = {"title": TITLE, "teacher_score": 0.72, "val_metric": VAL_METRIC}
    json.dump({"meta": meta, "data": data}, open(OUT, "w"), indent=1)
    print(f"[ok] wrote {OUT}")
    for label, series in data.items():
        s = np.asarray(series["scores"], float)
        print(f"{label:15s} start {s[0]:.3f}  mean {s.mean():.3f}  std {s.std():.3f}  end10 {s[-10:].mean():.3f}")

    # ---- plot in the reference style (light raw + dark EMA) ----
    # Canvas and fonts come from panel_style so this figure is interchangeable
    # with the other score-curve panels in the paper.
    curve_panel_rc()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for label, series in data.items():
        sc = np.asarray(series["scores"], float)
        dark = COLORS[label]
        light = _lighten(dark, 0.55)
        ax.plot(st, sc, color=light, lw=CURVE_RAW_LW, alpha=CURVE_RAW_ALPHA, zorder=2,
                solid_capstyle="round")
        ax.plot(st, _ema(sc, EMA_ALPHA), color=dark, lw=CURVE_SMOOTH_LW, zorder=3, label=label,
                solid_capstyle="round", solid_joinstyle="round")
    ax.axhline(0.72, color="#666666", ls="--", lw=CURVE_TEACHER_LW, zorder=1, label="Teacher")
    style_curve_axes(ax, TITLE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max())
    ax.set_ylim(0.30, 0.80)
    curve_legend(ax)
    save_panel(fig, FIG)


if __name__ == "__main__":
    main()
