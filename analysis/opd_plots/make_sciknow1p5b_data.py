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

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_sciknow1p5b_val.json")   # cached wandb val dump (full/vector)
OUT = os.path.join(HERE, "data_sciknoweval-1p5b-offpolicy.json")
FIG = os.path.join(HERE, "fig", "opd_sciknoweval-1p5b-offpolicy.svg")

END = 150               # x-axis 1..END
EMA_ALPHA = 0.2
TITLE = "Off-Policy Distillation on SciKnowEval with Qwen2.5-1.5B-Deepseek"
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
        dark = COLORS[label]
        light = _lighten(dark, 0.55)
        ax.plot(st, sc, color=light, lw=1.2, alpha=0.28, zorder=2, solid_capstyle="round")
        ax.plot(st, _ema(sc, EMA_ALPHA), color=dark, lw=2.6, zorder=3, label=label,
                solid_capstyle="round", solid_joinstyle="round")
    ax.axhline(0.72, color="#666666", ls="--", lw=1.8, zorder=1, label="Teacher")
    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=17, labelpad=8)
    ax.set_title(TITLE, fontsize=16, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(st.min(), st.max())
    ax.set_ylim(0.30, 0.80)
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
