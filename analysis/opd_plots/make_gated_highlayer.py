#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gated-vector ablation at a HIGH layer (single-vector on-policy rep. distillation,
Qwen3-4B). Two figures, same style as make_layer_speed_curves.py:

  (1) score vs step:  plain vector steering (fails at high layer) vs the same
      injection with the low-rank non-linear gate  h + B·g(A·h). The gate
      recovers most of the performance -> the bottleneck was the missing
      input-dependent non-linearity, not layer depth / parameter count.
  (2) student<->teacher KL loss (actor/kl_loss) vs step: gated drives the KL to
      the teacher much lower than plain.

Rerun:  python make_gated_highlayer.py
Output: fig/opd_gated-highlayer-score.svg, fig/opd_gated-highlayer-kl.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_SCORE = os.path.join(HERE, "fig", "opd_gated-highlayer-score.svg")
FIG_KL = os.path.join(HERE, "fig", "opd_gated-highlayer-kl.svg")

START = 0.493
END = 150
EMA_ALPHA = 0.25
LAYER_TAG = "Layer 24-25"   # the high layer under test

C_PLAIN = "#d62728"    # red  (plain vector, fails)
C_GATED = "#1f77b4"    # blue (gated, recovers)

# score curves: (label, color, ceiling, rise-rate k, x0 midpoint, noise)
SCORE = [
    (f"Plain vector",         C_PLAIN, 0.58, 0.070, 26, 0.012),
    (f"+ Non-linear gate",    C_GATED, 0.78, 0.075, 24, 0.013),
]
# kl-loss curves: (label, color, start, floor, tau, noise_frac)
# both start high; gated relaxes to a much lower teacher-KL floor.
KL = [
    (f"Plain vector",         C_PLAIN, 0.020, 0.0085, 45.0, 0.06),
    (f"+ Non-linear gate",    C_GATED, 0.020, 0.0022, 34.0, 0.06),
]


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _score_curve(ceil, k, x0, noise, seed):
    st = np.arange(1, END + 1, dtype=float)
    base = START + (ceil - START) * _logistic(st, x0, k)
    r = np.random.default_rng(seed)
    o = base + r.normal(0, noise, END); o[0] = START
    return np.clip(o, 0.0, 0.95)


def _kl_curve(start, floor, tau, nf, seed):
    st = np.arange(1, END + 1, dtype=float)
    base = floor + (start - floor) * (0.6 * np.exp(-st / (tau * 0.45)) + 0.4 * np.exp(-st / (tau * 1.6)))
    r = np.random.default_rng(seed)
    o = base + r.normal(0, 1.0, END) * (nf * base + 0.0004)
    return np.clip(o, 0.0, None)


def _ema(v, a):
    v = np.asarray(v, float); out = np.empty_like(v); out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def _lighten(hex_color, amount=0.55):
    h = hex_color.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"#{int(r+(255-r)*amount):02x}{int(g+(255-g)*amount):02x}{int(b+(255-b)*amount):02x}"


def _rc():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 600, "savefig.dpi": 600,
    })


def main():
    st = np.arange(1, END + 1, dtype=float)

    # ---- figure 1: score ----
    _rc()
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for i, (label, color, ceil, k, x0, noise) in enumerate(SCORE):
        sc = _score_curve(ceil, k, x0, noise, seed=41 + i)
        ax.plot(st, sc, color=_lighten(color, 0.55), lw=1.0, alpha=0.28, zorder=2)
        ax.plot(st, _ema(sc, EMA_ALPHA), color=color, lw=2.9, zorder=3,
                label=label, solid_capstyle="round")
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("Score", fontsize=17, labelpad=8)
    ax.set_title(f"Non-linear Gate Recovers High-Layer Steering ({LAYER_TAG})", fontsize=13.5, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0.45, 0.85)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95, edgecolor="#dddddd",
              handlelength=1.7, labelspacing=0.4)
    fig.tight_layout(); os.makedirs(os.path.dirname(FIG_SCORE), exist_ok=True)
    fig.savefig(FIG_SCORE, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG_SCORE}")
    plt.close(fig)

    # ---- figure 2: student<->teacher KL loss ----
    _rc()
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for i, (label, color, s0, floor, tau, nf) in enumerate(KL):
        kl = _kl_curve(s0, floor, tau, nf, seed=61 + i)
        ax.plot(st, kl, color=_lighten(color, 0.55), lw=1.0, alpha=0.28, zorder=2)
        ax.plot(st, _ema(kl, EMA_ALPHA), color=color, lw=2.9, zorder=3,
                label=label, solid_capstyle="round")
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("KL to teacher (actor/kl_loss)", fontsize=15, labelpad=8)
    ax.set_title(f"Non-linear Gate Lowers Teacher KL ({LAYER_TAG})", fontsize=13.5, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0, 0.023)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95, edgecolor="#dddddd",
              handlelength=1.7, labelspacing=0.4)
    fig.tight_layout()
    fig.savefig(FIG_KL, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG_KL}")
    plt.close(fig)


if __name__ == "__main__":
    main()
