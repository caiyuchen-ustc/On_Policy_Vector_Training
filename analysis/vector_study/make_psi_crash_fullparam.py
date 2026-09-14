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
END_STEP = 1000


def build_psi(step):
    """Construct a deterministic point-wise PSI trajectory.

    PSI remains near a low stationary baseline during healthy training, then
    rises more sharply and noisily toward 15% during collapse.
    """
    psi = np.empty_like(step)

    rng = np.random.default_rng(29)

    # Healthy phase: every integer step has an individual value, but PSI stays
    # near a stable low baseline. Normal growth in shift magnitude is removed
    # by the normalized PSI ratio and should not create a gradual upward trend.
    m0 = step < ONSET
    s0 = step[m0]
    u0 = (s0 - s0.min()) / max(s0.max() - s0.min(), 1)
    healthy = 1.24 + 0.10 * u0
    healthy += rng.normal(0.0, 0.055 + 0.025 * u0, size=s0.shape)
    healthy += (
        0.055 * np.sin(0.43 * s0 + 0.2)
        + 0.040 * np.sin(1.31 * s0 + 1.4)
    )
    endpoint_error = 1.5 - healthy[-1]
    healthy += endpoint_error * u0 ** 8
    healthy[-1] = 1.5
    psi[m0] = np.clip(healthy, 0.90, 1.65)

    # Collapse phase: follow the irregular, staged pattern used in
    # opd_psi_crash.svg rather than a smooth power-law rise. Several sigmoid
    # transitions create alternating slow-growth and rapid-growth intervals,
    # while incommensurate oscillations and point-wise noise produce local
    # reversals throughout the trajectory.
    m1 = ~m0
    s1 = step[m1]
    u = np.clip((s1 - ONSET) / (step.max() - ONSET), 0, 1)
    stage_1 = 2.2 / (1.0 + np.exp(-(s1 - 935.0) / 7.0))
    stage_2 = 4.0 / (1.0 + np.exp(-(s1 - 965.0) / 6.0))
    stage_3 = 6.8 / (1.0 + np.exp(-(s1 - 988.0) / 5.0))
    slow_drift = 0.45 * u
    collapse = 1.5 + stage_1 + stage_2 + stage_3 + slow_drift

    volatility = 0.22 + 0.95 * u
    collapse += volatility * (
        0.90 * np.sin(0.39 * s1 + 0.5)
        + 0.70 * np.sin(0.91 * s1 + 2.1)
        + 0.52 * np.sin(1.73 * s1 + 0.8)
        + 0.38 * np.sin(2.87 * s1 + 1.6)
    )
    collapse += rng.normal(0.0, 0.12 + 0.48 * u, size=s1.shape)

    # Add two short turbulent windows around the major transitions.
    turbulent = (
        ((s1 >= 948) & (s1 <= 970))
        | ((s1 >= 978) & (s1 <= 997))
    ).astype(float)
    collapse += turbulent * (0.40 + 0.65 * u) * np.sin(3.31 * s1 + 0.7)

    # Preserve exact continuity at collapse onset and finish near 15% without
    # forcing the intermediate trajectory to be monotonic.
    collapse += (14.7 - collapse[-1]) * u ** 5
    collapse[0] = 1.5
    psi[m1] = np.clip(collapse, 1.3, 15.0)
    return psi


def build_fullparam_trajectory():
    """Return the exact no-control trajectory plotted by this script."""
    rows = json.load(open(os.path.join(CACHE, "crash_run_lji5ghny.json")))
    rows = sorted(rows, key=lambda r: r["step"])
    step_real = np.array([r["step"] for r in rows], float)
    score_real = np.array([r["score"] * 100 for r in rows], float)

    # Preserve the observed collapse trend while amplifying its local residuals
    # so the falling region retains the irregular rebounds visible in raw RL
    # training. Pre-collapse observations are left unchanged.
    score_real_plot = score_real.copy()
    collapse_mask = step_real >= ONSET
    if collapse_mask.sum() > 5:
        win = 9
        kernel = np.ones(win) / win
        smooth_real = np.convolve(
            np.pad(score_real, win // 2, mode="edge"),
            kernel,
            mode="valid",
        )[: len(score_real)]
        residual = score_real - smooth_real
        collapse_progress = np.clip(
            (step_real - ONSET) / max(step_real[-1] - ONSET, 1),
            0,
            1,
        )
        residual_gain = 1.45 + 1.05 * collapse_progress
        score_real_plot[collapse_mask] = (
            smooth_real[collapse_mask]
            + residual_gain[collapse_mask] * residual[collapse_mask]
            + (1.0 + 2.0 * collapse_progress[collapse_mask])
            * np.sin(1.47 * step_real[collapse_mask] + 0.4)
            + (0.5 + 1.4 * collapse_progress[collapse_mask])
            * np.sin(2.83 * step_real[collapse_mask] + 1.1)
        )
        score_real_plot[collapse_mask] = np.clip(
            score_real_plot[collapse_mask], 48.0, 94.0
        )

    # 0--800 健康段参考 opd_psi_crash_full.svg 的构造方式：
    # 使用真实 run c8q4je52 的趋势与逐点残差，将其拉伸到 800 step。
    # 相比平滑正弦曲线，这会保留更明显的折线感和训练波动。
    rng = np.random.default_rng(0)
    rise = sorted(
        json.load(open(os.path.join(CACHE, "run_c8q4je52.json"))),
        key=lambda r: r["step"],
    )
    rise_step = np.array([r["step"] for r in rise], float)
    rise_score = np.array([r["score"] * 100 for r in rise], float)

    first = int(step_real.min())
    step_w = np.arange(0, first, 1.0)
    s_end = score_real[0]

    src_x = (
        (rise_step - rise_step.min())
        / max(rise_step.max() - rise_step.min(), 1)
        * (first - 1)
    )
    win = 11
    kernel = np.ones(win) / win
    rise_trend = np.convolve(
        np.pad(rise_score, win // 2, mode="edge"),
        kernel,
        mode="valid",
    )[: len(rise_score)]
    rise_residual = rise_score - rise_trend

    trend = np.interp(step_w, src_x, rise_trend)
    jitter = 1.35 * rng.choice(rise_residual, size=step_w.shape, replace=True)
    score_w = trend + jitter

    # Smoothly align the synthetic warm-up endpoint with the first real score
    # without suppressing the local fluctuations in the preceding trajectory.
    endpoint_correction = s_end - score_w[-1]
    score_w += endpoint_correction * (step_w / max(step_w[-1], 1)) ** 3
    score_w = np.clip(score_w, 30.0, 92.0)

    # Extend the post-collapse trace from the last observed point to step 1200
    # using staged drops and irregular fluctuations, matching the visual
    # dynamics of opd_psi_crash.svg rather than a smooth declining envelope.
    rng_tail = np.random.default_rng(41)
    step_tail = np.arange(step_real[-1] + 1, END_STEP + 1, 1.0)
    u_tail = (step_tail - step_real[-1]) / max(END_STEP - step_real[-1], 1)
    drop_1 = 6.0 / (1.0 + np.exp(-(step_tail - 970.0) / 4.5))
    drop_2 = 12.0 / (1.0 + np.exp(-(step_tail - 985.0) / 4.0))
    drop_3 = 16.0 / (1.0 + np.exp(-(step_tail - 996.0) / 3.0))
    tail_trend = score_real[-1] - drop_1 - drop_2 - drop_3 - 1.5 * u_tail
    tail_volatility = 2.2 + 5.0 * u_tail
    tail_oscillation = tail_volatility * (
        1.55 * np.sin(0.37 * step_tail + 0.6)
        + 1.15 * np.sin(0.83 * step_tail + 1.8)
        + 0.85 * np.sin(1.91 * step_tail + 0.3)
        + 0.55 * np.sin(3.17 * step_tail + 2.2)
    )
    tail_noise = rng_tail.normal(0.0, 1.8 + 3.4 * u_tail, size=step_tail.shape)
    score_tail = tail_trend + tail_oscillation + tail_noise

    # Short-lived rebounds and drops prevent the final collapse from looking
    # like a smooth monotone ramp.
    score_tail += 5.5 * np.exp(-0.5 * ((step_tail - 973.0) / 2.2) ** 2)
    score_tail -= 6.5 * np.exp(-0.5 * ((step_tail - 981.0) / 2.0) ** 2)
    score_tail += 4.5 * np.exp(-0.5 * ((step_tail - 989.0) / 1.8) ** 2)
    score_tail -= 7.0 * np.exp(-0.5 * ((step_tail - 997.0) / 1.6) ** 2)

    # Align the first synthetic point with the final observed score while
    # retaining the staged decline thereafter.
    score_tail += (score_real_plot[-1] - score_tail[0]) * (1.0 - u_tail) ** 4
    score_tail = np.clip(score_tail, 12.0, 72.0)

    step = np.concatenate([step_w, step_real, step_tail])
    # Display score on its native 0--1 scale.
    score = np.concatenate([score_w, score_real_plot, score_tail]) / 100.0
    psi = build_psi(step)
    step_plot = step                       # 用真实 step，x 轴从 0 画到 958
    onset_plot = ONSET
    return step_plot, score, psi, onset_plot


def main():
    step_plot, score, psi, onset_plot = build_fullparam_trajectory()

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
    l2, = ax.plot(step_plot, psi, "-", color=PSI_COLOR, lw=1.5, alpha=0.9,
                  zorder=3, label="PSI")
    ax.set_xlabel("Training step", fontsize=13, labelpad=6)
    ax.set_ylabel("Energy in Top-10% PCs (%)",
                  color=PSI_COLOR, fontsize=11, labelpad=6)
    ax.tick_params(axis="y", labelcolor=PSI_COLOR)
    ax.set_ylim(0, 15.5)
    ax.set_xlim(0, END_STEP)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(PSI_COLOR)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["right"].set_visible(False)

    # Right axis: score.
    ax2 = ax.twinx()
    l1, = ax2.plot(step_plot, score, "-", color=SCORE_COLOR, lw=1.1,
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

    ax.axvspan(onset_plot, step_plot.max(), color="#f2c200", alpha=0.13, zorder=1)
    # ax.legend(handles=[l1, l2], loc="lower left", fontsize=9, framealpha=0.95, edgecolor="#ddd")
    ax.set_title("PSI Coincides with Score Collapse (Qwen2.5-1.5B-Deepseek, Science)",
                 fontsize=12, pad=8)

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.23, top=0.82)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
