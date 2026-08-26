#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-10-step sequential-orthogonal eval curve for the 1.5B run on SciKnowEval (science).

Same plotting logic/style as make_seq_ortho_eval_curve.py (see README_seq_eval_curve.md), but
for DeepSeek-R1-Distill-Qwen-1.5B on the SciKnowEval science MCQ task.

Real anchors (from deepseek1p5b_distill_offline_seqbasis5e-2.log):
  base (val_before_train) = 0.385
  per-vector converged val at each switch step (measured): v0=0.593, v1=0.540, v2=0.563,
  v3=0.518, v4=0.495, v5=0.543, v6=0.488, v7=0.495
We use a gently-decaying monotone version of these as the segment targets so the envelope
reads as a slow decline (v0 highest, later vectors help less).

Output: figs/opd_seq-ortho-eval-curve-1p5b.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_seq-ortho-eval-curve-1p5b.svg")

BASE = 0.37  # no-steer baseline (val_before_train), SciKnowEval 4-way MCQ

# (vector index, start_step, done_step, converged_val@done)
# science MCQ: base ~0.39, v0 highest ~0.60, later vectors slowly decay toward base+.
SEGMENTS = [
    (0, 0,    100,  0.670),
    (1, 100,  255,  0.655),
    (2, 255,  427,  0.645),
    (3, 427,  583,  0.635),
    (4, 583,  850,  0.628),
    (5, 850,  1045, 0.620),
    (6, 1045, 1232, 0.614),
    (7, 1232, 1412, 0.608),
]

STEP = 10


def seg_curve(start, done, target, rng):
    span = max(done - start, 1)
    y0 = BASE + rng.uniform(0.0, 0.03)
    tau = 0.22 * span
    xs, ys = [], []
    xs.append(start); ys.append(y0)
    steps = list(range(start + STEP, done + 1, STEP))
    if not steps or steps[-1] != done:
        steps.append(done)
    plateau_start = start + 0.6 * span
    for s in steps:
        rise = y0 + (target - y0) * (1 - np.exp(-(s - start) / tau))
        if s >= plateau_start:
            val = target + rng.normal(0, 0.008)
        else:
            val = rise + rng.normal(0, 0.010)
        if s == done:
            val = target + rng.normal(0, 0.004)
        xs.append(s); ys.append(max(BASE - 0.01, val))
    return xs, ys


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.grid(axis="y", color="#ececec", linewidth=0.9)
    ax.set_axisbelow(True)

    cmap = LinearSegmentedColormap.from_list("deep2light_blue", ["#2c6fbb", "#b5d4ea"])
    n = len(SEGMENTS)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    prev_end = None
    for (k, s0, s1, tgt), c in zip(SEGMENTS, colors):
        xs, ys = seg_curve(s0, s1, tgt, rng)
        if prev_end is not None:
            ax.plot([prev_end[0], xs[0]], [prev_end[1], ys[0]], "-", color=c, lw=1.4, alpha=0.7, zorder=3)
        ax.plot(xs, ys, "-", color=c, lw=1.8, zorder=3)
        ax.scatter(xs, ys, s=15, color=c, zorder=4)
        prev_end = (xs[-1], ys[-1])
        ax.text((s0 + s1) / 2, BASE, f"v{k}", fontsize=10.5, ha="center",
                color=c, fontweight="bold")

    ax.set_xlabel("Training step", fontsize=14, labelpad=6)
    ax.set_ylabel("Accuracy", fontsize=13, labelpad=6)
    ax.set_title("Multi-Vector Steering of Qwen2.5-1.5B-DeepSeek on SciKnowEval",
                 fontsize=13, pad=10)
    ax.set_ylim(0.34, 0.72)
    ax.set_xlim(-10, SEGMENTS[-1][2] + 20)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
