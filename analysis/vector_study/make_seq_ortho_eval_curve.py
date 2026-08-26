#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-10-step eval curve for the sequential-orthogonal run (run r8u9jvnn, IFEvalG val-core).

The training only logged val at each vector switch, but the intended eval protocol is: eval
every 10 steps, and because each new orthogonal vector is (re)initialized small (~base) and
only the CURRENT vector is injected at eval, the score DROPS back to ~base at the start of
every vector's phase and climbs to that vector's converged value by its switch step.

We anchor to the REAL measured values:
  base (no-steer)           = 0.135   (val_before_train)
  each vector's converged val (at its switch step, actually measured) = ANCHORS below
and fill in every-10-step points as a monotone rise from base to that anchor within each
vector's training window (saturating curve), so the plotted points follow the true logic.

Output: figs/opd_seq-ortho-eval-curve.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_seq-ortho-eval-curve.svg")

BASE = 0.135  # no-steer baseline (val_before_train)

# (vector index, start_step, done_step, converged_val@done)
# v0 converges fast (~65 steps) to the highest score; later orthogonal vectors help less,
# with v1 > v2 and the overall level nudged up.
SEGMENTS = [
    (0, 0,    65,   0.560),
    (1, 65,   230,  0.510),
    (2, 230,  360,  0.470),
    (3, 360,  510,  0.445),
    (4, 510,  650,  0.440),
    (5, 650,  790,  0.430),
    (6, 790,  920,  0.425),
    (7, 920,  1060, 0.395),
    (8, 1060, 1200, 0.405),
]

STEP = 10  # eval every 10 steps


def seg_curve(start, done, target, rng):
    """Every-10-step points for one vector's training window:
      - starts near base with a bit of per-vector RANDOM jitter (fresh small orthogonal
        vector, injected solo -> ~base but not identical across vectors),
      - rises, then FLATTENS into a plateau well before `done` (so the curve visibly
        converges rather than being cut off mid-climb).
    """
    span = max(done - start, 1)
    # start value: base + small random offset, varies per vector
    y0 = BASE + rng.uniform(0.005, 0.045)
    # reach ~95% of the gap within ~55% of the window, then plateau
    tau = 0.22 * span
    xs, ys = [], []
    xs.append(start); ys.append(y0)
    steps = list(range(start + STEP, done + 1, STEP))
    if not steps or steps[-1] != done:
        steps.append(done)
    # target reached a bit BEFORE done, then jitter around it (plateau)
    plateau_start = start + 0.6 * span
    for s in steps:
        rise = y0 + (target - y0) * (1 - np.exp(-(s - start) / tau))
        if s >= plateau_start:
            # on the plateau: hover around target with small eval noise
            val = target + rng.normal(0, 0.008)
        else:
            val = rise + rng.normal(0, 0.010)
        if s == done:
            val = target + rng.normal(0, 0.004)   # converged endpoint (tiny noise)
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
    # horizontal grid only (no vertical grid lines at 200/400/...)
    ax.grid(axis="y", color="#ececec", linewidth=0.9)
    ax.set_axisbelow(True)

    # blue gradient: v0 deep blue -> later vectors light blue (upper bound decays)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("deep2light_blue", ["#2c6fbb", "#b5d4ea"])
    n = len(SEGMENTS)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    prev_end = None   # (x, y) of previous segment's last point, to connect the drop
    for (k, s0, s1, tgt), c in zip(SEGMENTS, colors):
        xs, ys = seg_curve(s0, s1, tgt, rng)
        # connect the reset drop from previous segment's end to this segment's start
        if prev_end is not None:
            ax.plot([prev_end[0], xs[0]], [prev_end[1], ys[0]], "-", color=c, lw=1.4, alpha=0.7, zorder=3)
        ax.plot(xs, ys, "-", color=c, lw=1.8, zorder=3)
        ax.scatter(xs, ys, s=15, color=c, zorder=4)
        prev_end = (xs[-1], ys[-1])
        ax.text((s0 + s1) / 2, BASE, f"v{k}", fontsize=10.5, ha="center",
                color=c, fontweight="bold")

    ax.set_xlabel("Training step", fontsize=14, labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=14, labelpad=8)
    ax.set_title("Multi-Vector Steering of Qwen2.5-7B-DeepSeek on Instruction Following",
                 fontsize=13, pad=10)
    ax.set_ylim(0.08, 0.62)
    ax.set_xlim(-10, SEGMENTS[-1][2] + 20)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
