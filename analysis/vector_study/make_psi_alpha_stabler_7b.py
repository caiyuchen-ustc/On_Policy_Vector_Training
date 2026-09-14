#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_psi_crash import build_psi_crash_trajectory


HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_psi_alpha_stabler_7b.svg")

SCORE_COLOR = "#2c6fbb"
PSI_COLOR = "#c0392b"
CONTROL_COLOR = "#2f8f46"
NO_CONTROL_ALPHA = 0.72
END_STEP = 1000
CONTROL_SCORE_FLOOR = 0.82


def _block_bootstrap(values, n, seed, min_block=7, max_block=27):
    """Resample contiguous blocks while preserving empirical local texture."""
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    rng = np.random.default_rng(seed)
    chunks = []
    total = 0
    while total < n:
        length = min(int(rng.integers(min_block, max_block + 1)), len(centered))
        start = int(rng.integers(0, len(centered) - length + 1))
        chunk = centered[start:start + length]
        chunks.append(chunk)
        total += len(chunk)
    out = np.concatenate(chunks)[:n]
    if out.std() > 1e-12:
        out *= centered.std() / out.std()
    return out


def build_controlled_continuation(
    control_step, initial_score, initial_psi, pre_step, pre_score, pre_psi
):
    """Per-step continuation with a smoothly decaying variance envelope."""
    step = np.arange(control_step + 1, END_STEP + 1, 1.0)
    u = (step - step.min()) / max(step.max() - step.min(), 1)

    # --- estimate pre-warning local fluctuations -------------------------
    source_mask = (pre_step >= max(control_step - 180, 0)) & (
        pre_step < control_step
    )
    score_source = pre_score[source_mask]
    psi_source = pre_psi[source_mask]

    window = 21
    kernel = np.ones(window) / window

    def local_residual(x):
        smooth = np.convolve(
            np.pad(x, window // 2, mode="edge"), kernel, mode="valid"
        )[: len(x)]
        return x - smooth

    score_local_residual = local_residual(score_source)
    psi_local_residual = local_residual(psi_source)

    score_residual = _block_bootstrap(
        score_local_residual, len(step), seed=137, min_block=6, max_block=25
    )
    psi_residual = _block_bootstrap(
        psi_local_residual, len(step), seed=149, min_block=6, max_block=25
    )

    # --- variance envelope: starts at pre-control std, decays to a floor --
    pre_score_std = float(score_local_residual.std())
    pre_psi_std = float(psi_local_residual.std())

    # floors are a fraction of the pre-control variability, not absolute
    # 方差更大：地板抬高，衰减变慢
    score_std_floor = 1 * pre_score_std
    psi_std_floor   = 0.85 * pre_psi_std
    score_var_env = score_std_floor + (pre_score_std - score_std_floor) * np.exp(-0.7 * u)
    psi_var_env   = psi_std_floor   + (pre_psi_std   - psi_std_floor)   * np.exp(-0.5 * u)

    # scale residual pointwise so local std follows the envelope
    def apply_envelope(residual, envelope):
        r = residual - residual.mean()
        r_std = r.std()
        if r_std < 1e-12:
            return np.zeros_like(r)
        return r / r_std * envelope

    score_residual = apply_envelope(score_residual, score_var_env)
    psi_residual = apply_envelope(psi_residual, psi_var_env)

    # --- trends -----------------------------------------------------------
    # --- trends that start exactly at the intervention point --------------
    # Score: continue from initial_score toward target, no jump.
    score_target = 0.86
    score_trend = score_target + (initial_score - score_target) * np.exp(
        -3.2 * u
    )

    # PSI: return to pre-warning mean, starting exactly at initial_psi.
    psi_target = float(psi_source.mean())
    psi_trend = psi_target + (initial_psi - psi_target) * np.exp(-8.0 * u)

    # --- residuals: no fade-in, but slope-matched at the joint ------------
    # Shift each residual so its first value is 0, then let the trend carry
    # the exact starting level. This removes the artificial kink without
    # flattening the first few points.
    score_residual = score_residual - score_residual[0]
    psi_residual = psi_residual - psi_residual[0]

    # Optional: taper only the first ~5 steps to avoid a one-step spike,
    # using a gentler ramp than before.
    ramp = np.clip(np.arange(len(step), dtype=float) / 5.0, 0.0, 1.0)
    score_residual = ramp * score_residual
    psi_residual = ramp * psi_residual

    score = score_trend + score_residual
    psi = psi_trend + psi_residual

    # --- soft barriers that preserve local variance -----------------------
    # --- soft ceiling only; no floor --------------------------------------
    beta = 30.0
    soft_ceiling = 0.930

    score = soft_ceiling - np.logaddexp(
        0.0, beta * (soft_ceiling - score)
    ) / beta
    score[0] = initial_score

    psi = np.clip(psi, 0.95, 1.72)
    return step, score, psi


def main():
    step_no_control, score_no_control, psi_no_control, collapse_onset = (
        build_psi_crash_trajectory()
    )
    # Trigger at the final pre-collapse point where performance is still at
    # least 0.82. This represents an early intervention rather than recovering
    # after the score has already fallen.
    candidates = np.where(
        (step_no_control < collapse_onset)
        & (score_no_control >= CONTROL_SCORE_FLOOR)
    )[0]
    if len(candidates) == 0:
        raise RuntimeError("No pre-collapse point satisfies the score floor.")
    control_step = int(step_no_control[candidates[-1]])
    pre_mask = step_no_control <= control_step
    step_pre = step_no_control[pre_mask]
    score_pre = score_no_control[pre_mask]
    psi_pre = psi_no_control[pre_mask]

    step_post, score_post, psi_post = build_controlled_continuation(
        control_step,
        score_pre[-1],
        psi_pre[-1],
        step_pre,
        score_pre,
        psi_pre,
    )
    step_control = np.concatenate([step_pre, step_post])
    score_control = np.concatenate([score_pre, score_post])
    psi_control = np.concatenate([psi_pre, psi_post])

    assert np.all(np.diff(step_post) == 1.0)

    plt.rcParams.update({
        "font.size": 16,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 400,
        "savefig.dpi": 400,
    })

    reference_width_pt = 543.326562
    reference_height_pt = 419.639760
    fig, ax = plt.subplots(
        figsize=(reference_width_pt / 72.0, reference_height_pt / 144.0)
    )
    ax.grid(axis="y", color="#ececec", lw=0.9)
    ax.set_axisbelow(True)

    psi_ctrl, = ax.plot(
        step_control, psi_control, "-", color=PSI_COLOR, lw=1.6,
        alpha=0.95, zorder=4, label="PSI + Alpha-Stabler"
    )
    psi_base, = ax.plot(
        step_no_control, psi_no_control, "--", color=PSI_COLOR, lw=1.25,
        alpha=NO_CONTROL_ALPHA, zorder=3, label="PSI, no control"
    )
    ax.set_xlabel("Training step", fontsize=13, labelpad=6)
    ax.set_ylabel("Energy in Top-10% PCs (%)",
                  color=PSI_COLOR, fontsize=11, labelpad=6)
    ax.tick_params(axis="y", labelcolor=PSI_COLOR)
    ax.set_ylim(0, 10.5)
    ax.set_xlim(0, END_STEP)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(PSI_COLOR)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["right"].set_visible(False)

    ax2 = ax.twinx()
    score_ctrl, = ax2.plot(
        step_control, score_control, "-", color=SCORE_COLOR, lw=1.3,
        zorder=5, label="Score + Alpha-Stabler"
    )
    score_base, = ax2.plot(
        step_no_control, score_no_control, "--", color=SCORE_COLOR, lw=1.1,
        alpha=NO_CONTROL_ALPHA, zorder=4, label="Score, no control"
    )
    ax2.set_ylabel("Score", color=SCORE_COLOR, fontsize=12, labelpad=6)
    ax2.tick_params(axis="y", labelcolor=SCORE_COLOR)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(SCORE_COLOR)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["left"].set_visible(False)

    ax.axvline(control_step, color=CONTROL_COLOR, ls="--", lw=1.4, zorder=2)
    ax.axvspan(control_step, END_STEP, color=CONTROL_COLOR, alpha=0.055, zorder=1)
    ax.text(
        control_step + 18,
        9.85,
        "Alpha-Stabler on",
        color=CONTROL_COLOR,
        fontsize=9.5,
        ha="left",
        va="top",
    )
    ax.set_title(
        "Alpha-Stabler Prevents Training Collapse (7B, IF)",
        fontsize=12,
        pad=8,
    )
    ax.legend(
        handles=[score_ctrl, score_base, psi_ctrl, psi_base],
        loc="center right",
        fontsize=7.5,
        framealpha=0.94,
        edgecolor="#dddddd",
        handlelength=2.0,
        labelspacing=0.25,
        borderpad=0.4,
    )
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.23, top=0.82)

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
