#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot OPD score-mean curves for three methods (full-param OPD / LoRA OPD / representation OPD).

Each method gets:
  - a light, thin RAW line (per-step score),
  - a bold, dark SMOOTHED line on top.

Smoothing = EMA, then "peak protection": the smoothed value at each step is
    max(EMA_t, running_max_of_raw_up_to_t)
so the smoothed line never drops below the best score seen so far — the peak is
never shaved off, and the dark line is monotonically non-decreasing.

>>> Fill in your own data in the DATA dict below (steps + scores per method). <<<
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# ----------------------------------------------------------------------------
# 1) DATA — replace these with your real numbers.
#    Each entry: "label" -> {"steps": [...], "scores": [...]}.
#    steps / scores must be the same length. steps can be uneven.
# ----------------------------------------------------------------------------
DATA = {
    "Full-param OPD": {
        "steps": [],   # e.g. [0, 5, 10, 15, ...]
        "scores": [],  # e.g. [0.12, 0.20, 0.28, ...]
    },
    "LoRA OPD": {
        "steps": [],
        "scores": [],
    },
    "Representation OPD": {
        "steps": [],
        "scores": [],
    },
}

# ----------------------------------------------------------------------------
# 2) Style knobs
# ----------------------------------------------------------------------------
EMA_ALPHA = 0.1          # smoothing strength: smaller = smoother (0.05~0.2 typical)
# Peak-handling mode for the dark smoothed line:
#   "soft"   : EMA that keeps normal ups/downs (pretty), but rescaled so its GLOBAL max equals
#              the raw global max — the peak is never shaved, mid-curve stays smooth. (recommended)
#   "strict" : smoothed = max(EMA, running_max_of_raw) — monotonically non-decreasing, never drops
#              below the best-so-far. Guarantees "never lowers any peak" literally, but looks step-like.
#   "none"   : plain EMA, no peak handling.
PEAK_MODE = "soft"
RAW_ALPHA = 0.28         # opacity of the light raw line
RAW_LW = 1.2             # raw line width
SMOOTH_LW = 2.6          # smoothed line width
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opd_score_curves.png")

# A clean, high-contrast palette (dark = smoothed, light = raw is the same hue lightened).
COLORS = {
    "Full-param OPD":     "#1f77b4",  # blue
    "LoRA OPD":           "#d62728",  # red
    "Representation OPD": "#2ca02c",  # green
}


def ema(values, alpha):
    """Exponential moving average."""
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
    """Smooth the raw series while protecting the peak.

    mode:
      "soft"   : EMA (keeps natural ups/downs), then scale/shift so the smoothed GLOBAL max
                 equals the raw GLOBAL max. Uses a gentle additive lift at/around the argmax so
                 the peak is reached without flattening the whole curve. Pretty + peak preserved.
      "strict" : max(EMA, running_max_of_raw) — monotonically non-decreasing (step-like).
      "none"   : plain EMA.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    sm = ema(values, alpha)

    if mode == "strict":
        running_max = np.maximum.accumulate(values)
        return np.maximum(sm, running_max)

    if mode == "soft":
        raw_peak = float(np.max(values))
        sm_peak = float(np.max(sm))
        if sm_peak < raw_peak:
            # lift only the deficit at the smoothed argmax, tapering to 0 elsewhere via a
            # normalized bump = sm/sm_peak, so the global max hits raw_peak but low regions
            # (early training) stay put. Keeps the curve smooth, just un-shaves the top.
            deficit = raw_peak - sm_peak
            weight = sm / sm_peak if sm_peak > 0 else np.zeros_like(sm)
            sm = sm + deficit * weight
        return sm

    return sm  # "none"


def _lighten(hex_color, amount=0.55):
    """Blend a hex color toward white by `amount` (0=orig, 1=white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    # aesthetics
    plt.rcParams.update({
        "font.size": 13,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.8,
        "figure.dpi": 130,
    })

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    any_data = False
    for label, series in DATA.items():
        steps = np.asarray(series.get("steps", []), dtype=float)
        scores = np.asarray(series.get("scores", []), dtype=float)
        if len(steps) == 0 or len(steps) != len(scores):
            print(f"[warn] '{label}': empty or mismatched steps/scores ({len(steps)} vs {len(scores)}), skipped")
            continue
        any_data = True

        # sort by step just in case
        order = np.argsort(steps)
        steps, scores = steps[order], scores[order]

        dark = COLORS.get(label, "#333333")
        light = _lighten(dark, 0.55)

        # light raw line (thin, translucent)
        ax.plot(steps, scores, color=light, lw=RAW_LW, alpha=RAW_ALPHA, zorder=2,
                solid_capstyle="round")

        # dark smoothed line on top
        sm = smooth_with_peak_protection(scores, EMA_ALPHA, PEAK_MODE)
        ax.plot(steps, sm, color=dark, lw=SMOOTH_LW, label=label, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")

    if not any_data:
        print("[error] No data filled in. Edit the DATA dict at the top of this script.")
        return

    ax.set_xlabel("Training step", fontsize=14, labelpad=8)
    ax.set_ylabel("Score (mean)", fontsize=14, labelpad=8)
    ax.set_title("On-Policy Distillation: score vs. training step", fontsize=15, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0.01)

    # de-clutter: top/right spines off
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9,
                    edgecolor="#cccccc", fontsize=12)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".png", ".pdf"), bbox_inches="tight")  # vector version for papers
    print(f"[ok] saved: {OUT_PATH}")
    print(f"[ok] saved: {OUT_PATH.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
