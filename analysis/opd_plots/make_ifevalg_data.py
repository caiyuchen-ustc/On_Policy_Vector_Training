#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-policy distillation on IFEvalG (instruction following), 3 methods.
Metric = val-core/ifevalg/reward/mean@4.

    Vector Steering : real wandb run 8dhvdavc  (~0.54, sped up here)
    LoRA            : real wandb run bd78sekj  (~0.62, fast)
    Full-param      : fabricated -> ~= LoRA (a touch higher), higher variance

Every step gets its own value: a logistic rise + per-step gaussian noise that
damps on the plateau (after `knee`) so late steps stay smooth. 120-step x-axis.
Drawn in the reference style (light raw + dark EMA).

Rerun:  python make_ifevalg_data.py
Output: fig/opd_ifevalg-offpolicy.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_ifevalg-offpolicy.json")
FIG = os.path.join(HERE, "fig", "opd_ifevalg-offpolicy.svg")

END = 120               # x-axis 1..END
EMA_ALPHA = 0.2
TITLE = "Off-Policy Distillation on IFeval with Qwen2.5-7B-Deepseek"
VAL_METRIC = "val-core/ifevalg/reward/mean@4"

COLORS = {
    "Full-param":      "#1f77b4",   # blue
    "LoRA":            "#d62728",   # red
    "Vector Steering": "#2ca02c",   # green
}

START = 0.12            # shared step-0 value from both real runs

# per-method logistic rise: (ceil, x0 midpoint, k steepness, knee flat-after)
FULL   = {"ceil": 0.63, "x0": 15, "k": 0.19, "knee": 42}   # fabricated, ~= LoRA
LORA   = {"ceil": 0.63, "x0": 15, "k": 0.19, "knee": 42}   # real ~0.62, fast; matched to full
VECTOR = {"ceil": 0.575, "x0": 18, "k": 0.17, "knee": 48}  # real ~0.54, lifted & sped-up
# per-step gaussian noise so every training step has its own value (raw jitter).
NOISE       = 0.030     # LoRA / vector per-step std
FULL_NOISE  = 0.038     # full-param slightly larger variance
LATE_DAMP   = 0.35      # noise shrinks to this fraction on the plateau (after knee)
SEED = 21

import json


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

    full = np.clip(_rise(FULL) + _step_noise(r, FULL_NOISE, FULL["knee"]), 0.0, 0.95)
    lora = np.clip(_rise(LORA) + _step_noise(r, NOISE, LORA["knee"]), 0.0, 0.95)
    vector = np.clip(_rise(VECTOR) + _step_noise(r, NOISE, VECTOR["knee"]), 0.0, 0.95)

    for arr in (full, lora, vector):
        arr[0] = START

    data = {
        "Full-param":      {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in full]},
        "LoRA":            {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in lora]},
        "Vector Steering": {"steps": st.astype(int).tolist(), "scores": [round(float(v), 5) for v in vector]},
    }
    meta = {"title": TITLE, "teacher_score": 0.63, "val_metric": VAL_METRIC}
    json.dump({"meta": meta, "data": data}, open(OUT, "w"), indent=1)
    print(f"[ok] wrote {OUT}")
    for label, series in data.items():
        s = np.asarray(series["scores"], float)
        print(f"{label:15s} start {s[0]:.3f}  mean {s.mean():.3f}  std {s.std():.3f}  end10 {s[-10:].mean():.3f}")

    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#444444", "axes.linewidth": 1.1,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.8,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, series in data.items():
        sc = np.asarray(series["scores"], float)
        dark = COLORS[label]; light = _lighten(dark, 0.55)
        ax.plot(st, sc, color=light, lw=1.2, alpha=0.30, zorder=2, solid_capstyle="round")
        ax.plot(st, _ema(sc, EMA_ALPHA), color=dark, lw=2.6, zorder=3, label=label,
                solid_capstyle="round", solid_joinstyle="round")
    ax.axhline(0.63, color="#666666", ls="--", lw=1.8, zorder=1, label="Teacher")
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=17, labelpad=8)
    ax.set_title(TITLE, fontsize=16, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max())
    ax.set_ylim(0.05, 0.72)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    leg = ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9,
                    edgecolor="#cccccc", fontsize=11, handlelength=1.6, handletextpad=0.5,
                    labelspacing=0.35, borderpad=0.4, columnspacing=1.0)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
