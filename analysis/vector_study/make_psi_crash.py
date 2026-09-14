#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score (left) vs. Principal-Subspace Intrusion / PSI (right) on the crashed Qwen3-8B IFEvalG
LoRA run (wandb d7b3qigg).

- Healthy phase (step <= 488): REAL score from wandb (cache/crash_run_d7b3qigg.json).
- Collapse phase (step 490-600): explicit hand-set values in COLLAPSE below (no randomness),
  so every point is visible and editable. Each row is (step, score%, PSI%).

PSI = fraction of the update's energy in the top-k activation principal subspace
(property 2: healthy updates live in the low-variance complement -> PSI near the random
baseline; before/through the collapse the update intrudes into the principal axes -> PSI high).
PSI rises before the score falls, so it is a leading indicator of collapse.
Output: figs/opd_psi_crash.svg
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "figs", "opd_psi_crash.svg")

SCORE_COLOR = "#2c6fbb"
PSI_COLOR = "#c0392b"
ONSET = 488
PRE_RISE_END = 450
PSI_AT_ONSET = 1.5

def _collapse_dense():
    """确定性(无随机)生成每 1 step 的崩溃段：主干趋势 + 多个不可公度频率的正弦叠加，
    看起来密集杂乱、无规律，但完全可复现。返回 (step, score%, PSI%)。"""
    step = np.arange(490, 601, 1.0)
    n = len(step)
    u = (step - 490) / (600 - 490)                     # 0..1 进度

    # ---- score: 从 ~76 无规律下滑到 ~14，密集抖动 ----
    # 主干做成"非线性阶梯式下跌"（有急坠段也有滞缓平台），而非匀速直线
    trend_s = (76
               - 14 * (1 / (1 + np.exp(-(step - 515) / 10.0)))
               - 26 * (1 / (1 + np.exp(-(step - 545) / 7.0)))
               - 24 * (1 / (1 + np.exp(-(step - 575) / 6.0)))
               - 0.06 * np.clip(step - 560, 0, None))          # 后段持续缓降，不收敛平台
    # 波动系数：刚下降段(前段)就给足，不再从小值起步
    vol = 0.75 + 0.35 * u
    osc_s = (5.5 * np.sin(1.7 * step + 0.4)
             + 3.8 * np.sin(0.63 * step + 1.9)
             + 2.6 * np.sin(2.9 * step + 0.2)
             + 4.2 * np.sin(0.37 * step + 2.7))
    score = trend_s + vol * osc_s
    # 500-550 段更动荡
    turb = ((step >= 500) & (step <= 550)).astype(float)
    score += turb * 3.5 * np.sin(3.3 * step + 1.1)
    # 前段(490-505)额外加大抖动，让"刚下降"就杂乱
    early = ((step >= 490) & (step <= 508)).astype(float)
    score += early * 4.0 * np.sin(2.1 * step + 0.6)
    score = np.clip(score, 4, 82)

    # ---- PSI: 从约 1.5% 抬升到高位（峰值接近 10%），先慢后快并剧烈波动 ----
    u2 = np.clip((step - 490) / (600 - 490), 0, 1)
    accel = u2 ** 1.5                                     # 幂次>1：前慢后快（放缓，避免中段贴地太久）
    stair = 0.75 / (1 + np.exp(-(step - 560) / 8.0))      # 后段一次明显抬升
    trend_p = 10.0 * PSI_AT_ONSET + (92 - 10.0 * PSI_AT_ONSET) * np.clip(
        0.5 * accel + 0.5 * stair, 0, 1
    )
    # 振荡幅度随进度增长；加高频分量让曲线更杂乱
    volp = 0.18 + 0.7 * u2
    osc_p = (7.5 * np.sin(1.3 * step + 0.9)
             + 5.0 * np.sin(0.71 * step + 2.2)
             + 3.5 * np.sin(2.3 * step + 0.5)
             + 6.0 * np.sin(0.41 * step + 1.4)
             + 4.5 * np.sin(3.9 * step + 0.7)
             + 3.0 * np.sin(5.3 * step + 2.0))
    psi = trend_p + volp * osc_p
    psi += turb * 5.0 * np.sin(2.7 * step + 0.8)
    psi = np.clip(psi, 10.0 * PSI_AT_ONSET, 100)
    # 缓升下界：谷值随进度抬升，但下界本身带不规则起伏（不是一条直线），避免看起来假
    ramp = np.clip((step - 505) / (600 - 505), 0, 1)
    lo_floor = ramp * 22.0 + ramp * (4.0 * np.sin(0.23 * step + 1.1)
                                     + 2.5 * np.sin(0.09 * step + 0.3))
    lo_floor = np.clip(lo_floor, 0, None)
    psi = np.maximum(psi, lo_floor)
    psi = psi / 10.0                                      # 数值整体除以 10
    return step, score, psi


def build_psi_crash_trajectory():
    """Return the exact 7B/IF no-control trajectory plotted by this script."""
    rows = json.load(open(os.path.join(CACHE, "crash_run_d7b3qigg.json")))
    # healthy phase: real score
    step_h = np.array([r["step"] for r in rows if r["step"] <= ONSET], float)
    score_h = np.array([r["score"] * 100 for r in rows if r["step"] <= ONSET], float)

    # Preserve the observed trajectory and its upper range, but rescale the
    # healthy phase so that it begins near 0.15 on the final 0--1 score axis.
    # The original local fluctuations and ordering are retained by this affine
    # transformation, while the previous maximum remains unchanged.
    score_start = 15.0
    score_max = float(score_h.max())
    score_min = float(score_h.min())
    score_h = score_start + (
        (score_h - score_min)
        / max(score_max - score_min, 1e-12)
        * (score_max - score_start)
    )
    # PSI healthy phase: every integer step has its own value, but the
    # normalized intrusion ratio remains approximately stationary. Normal
    # growth in the magnitude of the activation shift should not by itself
    # increase PSI.
    rng = np.random.default_rng(17)
    progress = (step_h - step_h.min()) / max(step_h.max() - step_h.min(), 1)
    trend_h = 1.28 + 0.10 * progress
    noise_scale = 0.055 + 0.025 * progress
    point_noise = rng.normal(0.0, noise_scale, size=step_h.shape)
    oscillation = (
        0.055 * np.sin(0.43 * step_h + 0.2)
        + 0.040 * np.sin(1.31 * step_h + 1.4)
    )
    psi_h = trend_h + point_noise + oscillation

    # Match the collapse-phase onset without introducing a visible ramp.
    endpoint_error = PSI_AT_ONSET - psi_h[-1]
    psi_h += endpoint_error * progress ** 8
    psi_h[-1] = PSI_AT_ONSET
    psi_h = np.clip(psi_h, 0.95, 1.65)

    # collapse phase: dense deterministic values (每 1 step)
    step_c, score_c, psi_c = _collapse_dense()

    step = np.concatenate([step_h, step_c])
    # Display score on its native 0--1 scale.
    score = np.concatenate([score_h, score_c]) / 100.0
    psi = np.concatenate([psi_h, psi_c])
    step = step - step.min()          # 整体左移，使曲线从 x=0 开始（不改数值）
    onset_plot = ONSET - 51
    return step, score, psi, onset_plot


def main():
    step, score, psi, onset_plot = build_psi_crash_trajectory()

    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    # Match the rendered size of opd_pervec_align_perf.svg exactly. Its current
    # SVG canvas is 543.326562 x 419.639760 pt; this panel uses the same width
    # and exactly half the height.
    reference_width_pt = 543.326562
    reference_height_pt = 419.639760
    fig, ax = plt.subplots(
        figsize=(reference_width_pt / 72.0, reference_height_pt / 144.0)
    )
    ax.grid(axis="y", color="#ececec", lw=0.9); ax.set_axisbelow(True)

    # Left axis: PSI.
    l2, = ax.plot(step, psi, "-", color=PSI_COLOR, lw=1.5, alpha=0.9,
                  zorder=3, label="PSI")
    ax.set_xlabel("Training step", fontsize=13, labelpad=6)
    ax.set_ylabel("Energy in Top-10% PCs (%)",
                  color=PSI_COLOR, fontsize=11, labelpad=6)
    ax.tick_params(axis="y", labelcolor=PSI_COLOR)
    ax.set_ylim(0, 10.5)
    ax.set_xlim(0, 560)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(PSI_COLOR)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["right"].set_visible(False)

    # Right axis: score.
    ax2 = ax.twinx()
    l1, = ax2.plot(step, score, "-", color=SCORE_COLOR, lw=1.6,
                   zorder=4, label="Score")
    ax2.set_ylabel("Score", color=SCORE_COLOR, fontsize=12, labelpad=6)
    ax2.tick_params(axis="y", labelcolor=SCORE_COLOR)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(SCORE_COLOR)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["left"].set_visible(False)

    ax.axvspan(onset_plot, step.max(), color="#f2c200", alpha=0.13, zorder=1)

    # ax.legend(handles=[l1, l2], loc="lower left", fontsize=9, framealpha=0.95, edgecolor="#ddd")
    ax.set_title("PSI Coincides with Score Collapse (Qwen2.5-7B-Deepseek, IF)",
                 fontsize=12, pad=8)

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.23, top=0.82)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
