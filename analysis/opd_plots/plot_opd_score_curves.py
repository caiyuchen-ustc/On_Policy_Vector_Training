#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot OPD score-mean curves for three methods (Full-param / LoRA / Vector Steering).

Data lives in per-model JSON files next to this script, named data_<MODEL>.json, e.g.
    data_qwen2.5-1.5b-deepseek.json
    data_qwen3-4b.json
Each JSON has:
    {
      "meta": {"title": "...", "teacher_score": 0.855, "val_metric": "critic/score/mean"},
      "data": {
        "Full-param":      {"steps": [...], "scores": [...]},
        "LoRA":            {"steps": [...], "scores": [...]},
        "Vector Steering": {"steps": [...], "scores": [...]}
      }
    }

Each method gets a light thin RAW line + a bold dark SMOOTHED line on top.
The output SVG is written to fig/opd_<MODEL>.svg.

Usage:
    python plot_opd_score_curves.py qwen3-4b
    python plot_opd_score_curves.py qwen2.5-1.5b-deepseek
    python plot_opd_score_curves.py                 # defaults to DEFAULT_MODEL below
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from panel_style import (CURVE_RAW_ALPHA, CURVE_RAW_LW, CURVE_SMOOTH_LW,
                        CURVE_TEACHER_LW, PANEL_FIGSIZE, curve_legend,
                        curve_panel_rc, save_panel, style_curve_axes)

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Which model to plot: `python plot_opd_score_curves.py <model>` or edit this.
# <model> matches data_<model>.json in this folder.
# ----------------------------------------------------------------------------
DEFAULT_MODEL = "qwen3-4b"

# ----------------------------------------------------------------------------
# Style knobs
# ----------------------------------------------------------------------------
EMA_ALPHA = 0.2          # smoothing strength: smaller = smoother (0.05~0.2 typical)
# Peak-handling mode for the dark smoothed line:
#   "soft"   : EMA rescaled so its GLOBAL max equals the raw global max (peak never shaved).
#   "strict" : max(EMA, running_max_of_raw) — monotonically non-decreasing (step-like).
#   "none"   : plain EMA.
PEAK_MODE = "none"
# Line widths come from panel_style: they are tuned for the panel canvas,
# where 2.6 pt reads much heavier than it did on the old 7x5.5 figure.
RAW_ALPHA = CURVE_RAW_ALPHA
RAW_LW = CURVE_RAW_LW
SMOOTH_LW = CURVE_SMOOTH_LW

# A clean, high-contrast palette (dark = smoothed, light = raw is the same hue lightened).
COLORS = {
    "Full-param":      "#1f77b4",  # blue
    "LoRA":            "#d62728",  # red
    "Vector Steering": "#2ca02c",  # green
    "RL":              "#1f77b4",  # blue
    "SFT":             "#ff7f0e",  # orange
    "RL Teacher":      "#1f77b4",  # blue  (good)
    "SFT Teacher":     "#ff7f0e",  # orange (degrades)
    "Shifted Teacher": "#d62728",  # red   (collapses)
    # fig1: representation training, data-source / teacher variants
    # on-policy group (student self-generates)
    "On-policy + RL Teacher":               "#2ca02c",  # green
    "On-policy + SFT Teacher":              "#ff7f0e",  # orange
    "On-policy + Off-distribution Teacher": "#d62728",  # red
    # off-policy group (teacher generates)
    "Off-policy + RL Teacher":              "#1f77b4",  # blue
    "Off-policy + SFT Teacher":             "#9467bd",  # purple
    "Off-policy + Off-distribution Teacher":"#8c564b",  # brown
}


def ema(values, alpha):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    acc = values[0]
    out[0] = acc
    for i in range(1, len(values)):
        acc = alpha * values[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def smooth_with_peak_protection(values, alpha, mode="soft"):
    """Smooth the raw series while protecting the peak (see PEAK_MODE)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    sm = ema(values, alpha)
    if mode == "strict":
        return np.maximum(sm, np.maximum.accumulate(values))
    if mode == "soft":
        raw_peak = float(np.max(values))
        sm_peak = float(np.max(sm))
        if sm_peak < raw_peak:
            deficit = raw_peak - sm_peak
            weight = sm / sm_peak if sm_peak > 0 else np.zeros_like(sm)
            sm = sm + deficit * weight
        return sm
    return sm  # "none"


def _lighten(hex_color, amount=0.55):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def load_model(model):
    path = os.path.join(HERE, f"data_{model}.json")
    if not os.path.exists(path):
        avail = sorted(os.path.basename(p)[len("data_"):-len(".json")]
                       for p in glob.glob(os.path.join(HERE, "data_*.json")))
        raise SystemExit(f"[error] {path} not found. Available models: {avail}")
    with open(path) as f:
        blob = json.load(f)
    return blob.get("meta", {}), blob["data"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    meta, DATA = load_model(model)
    teacher_score = meta.get("teacher_score", None)
    title = meta.get("title", f"On-Policy Distillation ({model})")
    # filename encodes the smoothing so different settings don't overwrite each other:
    #   PEAK_MODE="none"  -> opd_<model>_ema<alpha>.svg
    #   PEAK_MODE="soft"  -> opd_<model>_soft-ema<alpha>.svg
    smooth_tag = f"ema{EMA_ALPHA:g}" if PEAK_MODE == "none" else f"{PEAK_MODE}-ema{EMA_ALPHA:g}"
    out_path = os.path.join(HERE, "fig", f"opd_{model}_{smooth_tag}.svg")

    # Canvas and fonts come from panel_style so all eight score-curve figures
    # (4 on-policy here + 4 off-policy) are interchangeable in the paper.
    curve_panel_rc()

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    any_data = False
    all_steps = []
    for label, series in DATA.items():
        steps = np.asarray(series.get("steps", []), dtype=float)
        scores = np.asarray(series.get("scores", []), dtype=float)
        if len(steps) == 0 or len(steps) != len(scores):
            print(f"[warn] '{label}': empty or mismatched steps/scores, skipped")
            continue
        any_data = True
        order = np.argsort(steps)
        steps, scores = steps[order], scores[order]
        all_steps += list(steps)

        dark = COLORS.get(label, "#333333")
        light = _lighten(dark, 0.55)
        ax.plot(steps, scores, color=light, lw=RAW_LW, alpha=RAW_ALPHA, zorder=2,
                solid_capstyle="round")
        sm = smooth_with_peak_protection(scores, EMA_ALPHA, PEAK_MODE)
        ax.plot(steps, sm, color=dark, lw=SMOOTH_LW, label=label, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")

    if not any_data:
        raise SystemExit(f"[error] no usable data in data_{model}.json")

    if teacher_score is not None:
        ax.axhline(teacher_score, color="#666666", ls="--", lw=CURVE_TEACHER_LW, zorder=1, label="RL Teacher")

    style_curve_axes(ax, title, ylabel="Score")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0)
    if all_steps:
        ax.set_xlim(left=min(all_steps), right=max(all_steps))

    curve_legend(ax)

    save_panel(fig, out_path)


if __name__ == "__main__":
    main()
