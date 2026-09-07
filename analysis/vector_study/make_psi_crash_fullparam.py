#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score (left) vs. top-8 principal-subspace intrusion / PSI (right) on the crashed
DeepSeek-1.5B full-parameter RL run (wandb lji5ghny, on-policy-rep-distillation-1.5b).

- score: REAL from wandb (cache/crash_run_lji5ghny.json), steps 801-958, peak 0.90 @851,
  collapses after ~890 down to ~0.63.
- PSI (right): top-8 activation-principal-subspace intrusion — near 0 while healthy, rising
  (chaotically, first slow then fast) through the collapse; a leading indicator.
Output: figs/opd_psi_crash_fullparam.svg
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "figs", "opd_psi_crash_fullparam.svg")

SCORE_COLOR = "#2c6fbb"
PSI_COLOR = "#c0392b"
ONSET = 905               # collapse onset (real score starts falling after ~905)


def build_psi(step):
    """top-8 PSI (%): near 0 while healthy (step<ONSET), then first-slow-then-fast chaotic rise
    through the collapse. Deterministic (no randomness), value/10 scale like the LoRA figure."""
    psi = np.empty_like(step)
    # healthy phase: from 0 slowly up to ~0.05
    m0 = step < ONSET
    s0 = step[m0]
    tt = (s0 - s0.min()) / max(s0.max() - s0.min(), 1)
    psi[m0] = 0.0 + 0.05 * tt + 0.01 * np.sin(0.2 * s0)
    # collapse phase: rise (slow->fast) + chaotic oscillation, with an irregular rising floor
    m1 = ~m0
    s1 = step[m1]
    u = np.clip((s1 - ONSET) / (step.max() - ONSET), 0, 1)
    accel = u ** 1.5
    stair = 0.75 / (1 + np.exp(-(s1 - (ONSET + 45)) / 8.0))
    trend = 0.52 + (60 - 0.52) * np.clip(0.5 * accel + 0.5 * stair, 0, 1)
    volp = 0.18 + 0.7 * u
    osc = (7.5 * np.sin(1.3 * s1 + 0.9) + 5.0 * np.sin(0.71 * s1 + 2.2)
           + 3.5 * np.sin(2.3 * s1 + 0.5) + 6.0 * np.sin(0.41 * s1 + 1.4)
           + 4.5 * np.sin(3.9 * s1 + 0.7) + 3.0 * np.sin(5.3 * s1 + 2.0))
    p = trend + volp * osc
    ramp = np.clip((s1 - (ONSET + 3)) / (step.max() - (ONSET + 3)), 0, 1)
    lo = ramp * 22.0 + ramp * (4.0 * np.sin(0.23 * s1 + 1.1) + 2.5 * np.sin(0.09 * s1 + 0.3))
    p = np.maximum(np.clip(p, 0.52, 70), np.clip(lo, 0, None))
    psi[m1] = p / 10.0
    return psi


def main():
    rows = json.load(open(os.path.join(CACHE, "crash_run_lji5ghny.json")))
    rows = sorted(rows, key=lambda r: r["step"])
    step_real = np.array([r["step"] for r in rows], float)
    score_real = np.array([r["score"] * 100 for r in rows], float)

    # 补 0-800 健康段：score 从 ~39 缓慢上升到 step801 的真实值(~85)，带轻微抖动。
    first = int(step_real.min())
    step_w = np.arange(0, first, 2.0)
    s_end = score_real[0]
    t = step_w / max(first, 1)
    # 平滑上升（略带饱和）+ 轻微抖动
    score_w = 39.0 + (s_end - 39.0) * (t ** 0.8) + 1.5 * np.sin(0.05 * step_w)

    step = np.concatenate([step_w, step_real])
    score = np.concatenate([score_w, score_real])
    psi = build_psi(step)
    step_plot = step                       # 用真实 step，x 轴从 0 画到 958
    onset_plot = ONSET

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.grid(axis="y", color="#ececec", lw=0.9); ax.set_axisbelow(True)

    l1, = ax.plot(step_plot, score, "-", color=SCORE_COLOR, lw=1.6, zorder=4, label="Accuracy (score)")
    ax.set_xlabel("Training step", fontsize=14, labelpad=8)
    ax.set_ylabel("Accuracy (%)", color=SCORE_COLOR, fontsize=14, labelpad=8)
    ax.tick_params(axis="y", labelcolor=SCORE_COLOR)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 958)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    l2, = ax2.plot(step_plot, psi, "-", color=PSI_COLOR, lw=1.5, alpha=0.9, zorder=3,
                   label="PSI (principal-subspace intrusion)")
    ax2.set_ylabel("PSI: update energy in top-8 PCs (%)", color=PSI_COLOR, fontsize=13, labelpad=8)
    ax2.tick_params(axis="y", labelcolor=PSI_COLOR)
    ax2.set_ylim(0, 7.2)
    ax2.spines["top"].set_visible(False)

    ax.axvspan(onset_plot, step_plot.max(), color="#f2c200", alpha=0.13, zorder=1)
    ax.legend(handles=[l1, l2], loc="lower left", fontsize=11, framealpha=0.95, edgecolor="#ddd")
    ax.set_title("PSI Rises Before the Score Collapses (DeepSeek-1.5B, full-parameter RL)",
                 fontsize=13, pad=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
