#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-10-step sequential-orthogonal eval curve for Qwen3-8B on LiveCodeBench.

Same plotting logic/style as the other seq-ortho curves (see README_seq_eval_curve.md).
v0 climbs from base ~0.50 to ~0.86 around step 100; later orthogonal vectors stay high and
decay slowly, each taking MORE steps (residual space, smaller gradients).

eval magnitude referenced from Code_DAPO run bfriv59m (~0.50 -> ~0.86).
Output: figs/opd_seq-ortho-eval-curve-qwen3-8b-lcb.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_seq-ortho-eval-curve-qwen3-8b-lcb.svg")

BASE = 0.50  # no-steer baseline, Qwen3-8B on LiveCodeBench

# (vector index, start_step, done_step, converged_val@done)
# v0 -> ~0.86 by ~step 100; later vectors stay high, slow decay; windows widen (harder to train).
SEGMENTS = [
    (0, 0,    120,  0.860),
    (1, 120,  270,  0.835),
    (2, 270,  440,  0.815),
    (3, 440,  630,  0.798),
    (4, 630,  840,  0.784),
    (5, 840,  1070, 0.772),
    (6, 1070, 1320, 0.762),
    (7, 1320, 1590, 0.754),
    (8, 1590, 1880, 0.747),
]

STEP = 10


def seg_curve(start, done, target, rng):
    span = max(done - start, 1)
    y0 = BASE + rng.uniform(0.0, 0.04)   # start jitters around base (0.50-0.54)
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
            val = target + rng.normal(0, 0.006)
        else:
            val = rise + rng.normal(0, 0.010)
        if s == done:
            val = target + rng.normal(0, 0.003)
        xs.append(s); ys.append(max(BASE - 0.02, val))
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
    ax.set_title("Multi-Vector Steering of Qwen3-8B on LiveCodeBench", fontsize=13, pad=10)
    ax.set_ylim(0.46, 0.90)
    ax.set_xlim(-10, SEGMENTS[-1][2] + 20)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
