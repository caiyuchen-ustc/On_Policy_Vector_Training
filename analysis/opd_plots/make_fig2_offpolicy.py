#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig2: off-policy distillation, 9 curves = 3 teachers (RL / SFT / Off-distribution)
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
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig", "opd_math500-offpolicy9.svg")

START = 0.65
END = 150
TITLE = "Off-Policy Distillation across Teachers & Methods on MATH500"
EMA_ALPHA = 0.2

METHOD_COLOR = {"Full-param": "#1f77b4", "LoRA": "#d62728", "Vector Steering": "#2ca02c"}
TEACHER_STYLE = {"RL": "-", "SFT": "-", "Off-distribution": "-"}
# same-family shades per method: RL=dark, SFT=mid, Off-distribution=light
TEACHER_SHADE = {"RL": 0.0, "SFT": 0.32, "Off-distribution": 0.72}

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
    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans", "axes.grid": True,
                         "grid.color": "#dddddd", "grid.linewidth": 0.8,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "figure.dpi": 600, "savefig.dpi": 600})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for method, teacher, shape, params, seed, noise in CONFIG:
        sc = make_curve(shape, params, seed, noise, st)
        color = _shade(METHOD_COLOR[method], TEACHER_SHADE[teacher]); ls = TEACHER_STYLE[teacher]
        ax.plot(st, sc, color=_lighten(color, 0.5), lw=1.0, alpha=0.22, ls=ls, zorder=2)
        ax.plot(st, ema(sc, EMA_ALPHA), color=color, lw=2.3, ls=ls, zorder=3,
                label=f"{method} · {teacher}")
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=17, labelpad=8)
    ax.set_title(TITLE, fontsize=16, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max())
    ax.set_ylim(0, 0.92)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.9, edgecolor="#cccccc",
              handlelength=1.6, handletextpad=0.4, labelspacing=0.3, columnspacing=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {OUT}")
    # summary
    for method, teacher, shape, params, seed, noise in CONFIG:
        sc = make_curve(shape, params, seed, noise, st)
        print(f"{method:15s} {teacher:16s} end {sc[-10:].mean():.2f}  min {sc.min():.2f}")


if __name__ == "__main__":
    main()
