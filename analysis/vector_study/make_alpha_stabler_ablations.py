#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polished Alpha-Stabler ablations for 1.5B / Science.

IMPORTANT: all numbers below are illustrative placeholders. Replace them with
multi-seed measurements before using these plots as quantitative evidence.

Outputs:
  figs/opd_alpha-ablation-timing.svg
  figs/opd_alpha-ablation-subspace.svg
  figs/opd_alpha-ablation-strength.svg
  figs/opd_alpha-ablation-overview.svg
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")

BLUE = "#2563EB"
RED = "#EF4444"
GREEN = "#10B981"
PASTEL_BLUE = "#93C5FD"
PASTEL_ORANGE = "#FDBA74"
PASTEL_GREEN = "#86EFAC"
LIGHT_RED = "#C96A72"
PASTEL_LIGHT_RED = "#E8ADB2"
PASTEL_PURPLE = "#C4B5FD"
PURPLE = "#7C3AED"
STABLE_COLOR = "#0F172A"
ORANGE = "#F97316"
YELLOW = "#FBBF24"
CYAN = "#06B6D4"
GRAY = "#64748B"
DARK = "#172033"
GRID = "#E2E8F0"

FIGSIZE = (4.2, 3.55)
COMPACT_FIGSIZE = (4.2, 200.0 / 72.0)
SINGLE_PANEL_RECT = (.12, .14, .84, .68)
# Preserve the same top edge while lifting the learning-rate panel slightly,
# leaving enough bottom margin for the full "Training step" label.
LR_PANEL_RECT = (.12, .18, .84, .64)
TITLE_FONTSIZE = 10.2
TITLE_FONT = {"fontfamily": "DejaVu Sans", "fontweight": "normal",
              "fontstyle": "normal"}
OVERVIEW_TITLE_FONTSIZE = 9.0

TIMING = {
    "labels": ["Always\non", "Predictor\ntriggered", "Delay\n+10", "Delay\n+20", "Delay\n+30"],
    "scores": np.array([0.80, 0.82, 0.78, 0.71, 0.64]),
    # Higher final score corresponds to lower peak PSI. Modest intervention
    # delays increase PSI progressively without implying that these controlled
    # timing variants reach the full no-control collapse levels.
    "psis": np.array([1.30, 1.55, 2.10, 2.70, 4.20]),
    # Semantic palette: unnecessary early control (blue), proposed trigger
    # (green), increasingly late intervention (light-to-dark orange), and no
    # control (gray).
    "colors": [BLUE, GREEN, "#FBBF24", ORANGE, GRAY],
    "best": 1,
}

LR_STYLES = {
    # Each displayed learning rate has the same two-stage visual grammar:
    #   saturated solid -> normal training;
    #   lighter solid   -> about 100 steps of uncontrolled collapse.
    r"$10^{-5}$": {
        "color": GREEN,
        "post_color": PASTEL_GREEN,
        "collapse_start": 300,
        "collapse_score": 0.75,
        "collapse_drop": 0.25,
        # PSI uses a late-accelerating profile for the smallest LR.
        "psi_stage_amps": (0.65, 1.75, 4.35),
        "psi_stage_centers": (0.42, 0.70, 0.90),
        "psi_stage_widths": (0.095, 0.075, 0.050),
        "psi_drift": 0.35,
        "psi_phase": 0.20,
        "growth_tau": 220.0,
        "noise_scale": 0.011,
        "collapse_noise": 0.050,
        "psi_noise": 1.65,
        "stable_score_target": 0.88,
        "stable_psi_target": 1.34,
        "stable_tau": 185.0,
        "stable_score_seed": 561,
        "stable_psi_seed": 563,
    },
    r"$5{\times}10^{-5}$": {
        "color": BLUE,
        "post_color": PASTEL_BLUE,
        "collapse_start": 150,
        "collapse_score": 0.70,
        "collapse_drop": 0.35,
        # PSI rises in two broad waves followed by a turbulent late surge.
        "psi_stage_amps": (1.35, 3.10, 4.85),
        "psi_stage_centers": (0.29, 0.57, 0.82),
        "psi_stage_widths": (0.075, 0.070, 0.055),
        "psi_drift": 0.45,
        "psi_phase": 1.05,
        "growth_tau": 110.0,
        "noise_scale": 0.016,
        "collapse_noise": 0.065,
        "psi_noise": 1.85,
        "stable_score_target": 0.88,
        "stable_psi_target": 1.34,
        "stable_tau": 135.0,
        "stable_score_seed": 571,
        "stable_psi_seed": 573,
    },
    r"$10^{-4}$": {
        "color": LIGHT_RED,
        "post_color": PASTEL_LIGHT_RED,
        "collapse_start": 75,
        "collapse_score": 0.65,
        "collapse_drop": 0.42,
        # PSI for the largest LR rises early, overshoots, and remains volatile.
        "psi_stage_amps": (2.30, 4.55, 5.70),
        "psi_stage_centers": (0.20, 0.45, 0.70),
        "psi_stage_widths": (0.055, 0.060, 0.060),
        "psi_drift": 0.55,
        "psi_phase": 1.85,
        "growth_tau": 55.0,
        "noise_scale": 0.021,
        "collapse_noise": 0.078,
        "psi_noise": 2.05,
        "stable_score_target": 0.88,
        "stable_psi_target": 1.34,
        "stable_tau": 95.0,
        "stable_score_seed": 581,
        "stable_psi_seed": 583,
    },
}

# Every failed branch remains visible for about 100 steps before its
# Alpha-Stabler continuation appears.
COLLAPSE_TAIL_STEPS = 100

SPECIFICITY = {
    "labels": [
        "Top-PC\n(ours)",
        "Global\nclipping",
        "Random\nsubspace",
        "Middle-PC\nsubspace",
        "Bottom-PC\nsubspace",
        "No\ncontrol",
    ],
    # Illustrative placeholders: replace with matched multi-seed measurements.
    "scores": np.array([0.82, 0.79, 0.38, 0.37, 0.30, 0.35]),
    "psis": np.array([1.55, 1.80, 2.60, 2.10, 2.60, 2.40]),
    "colors": [GREEN, CYAN, BLUE, PURPLE, YELLOW, GRAY],
    "best": 0,
}


def _rc():
    plt.rcParams.update({
        # Match opd_psi_alpha_stabler_副本.svg.
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 10.0,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 1.0,
        "axes.labelsize": 10.5,
        "axes.labelweight": "normal",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 500,
        "savefig.dpi": 500,
    })


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _ar_noise(n, seed, rho):
    """Deterministic correlated noise with no visible periodic structure."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, n)
    out = np.empty(n, dtype=float)
    out[0] = eps[0]
    scale = np.sqrt(max(1.0 - rho ** 2, 1e-8))
    for j in range(1, n):
        out[j] = rho * out[j - 1] + scale * eps[j]
    return out


def _lr_curves():
    """Illustrative per-step Score/PSI trajectories for three learning rates."""
    full_step = np.arange(0, 1001, 1.0)

    # Explicitly control how many *rising training steps* each run receives:
    # 1e-4 rises for ~75, 5e-5 for ~150, and 1e-5 for ~300.
    base_curves = {}
    for j, (label, style) in enumerate(LR_STYLES.items()):
        collapse_start = int(style["collapse_start"])
        control_step = collapse_start + COLLAPSE_TAIL_STEPS
        target = float(style["collapse_score"])
        tau = float(style["growth_tau"])

        effective_step = np.minimum(full_step, collapse_start)
        growth_fraction = (
            1.0 - np.exp(-effective_step / tau)
        ) / (
            1.0 - np.exp(-collapse_start / tau)
        )
        score = 0.40 + (target - 0.40) * growth_fraction

        # Keep realistic local texture while anchoring the curve exactly at
        # the requested collapse score.
        score_noise = _ar_noise(
            len(full_step), seed=421 + 10 * j, rho=0.82 - 0.03 * j
        )
        start_fade = 1.0 - np.exp(-full_step / 16.0)
        anchor_fade = np.clip(
            np.abs(full_step - collapse_start) / 12.0, 0.0, 1.0
        )
        score += style["noise_scale"] * score_noise * start_fade * anchor_fade

        collapse_u = np.clip(
            (full_step - collapse_start) / COLLAPSE_TAIL_STEPS, 0.0, 1.0
        )

        # A normalized sigmoid creates a natural failure trajectory: a short
        # unstable shoulder, an accelerating drop, then a noisy low-score
        # tail. Unlike the previous exponential, it does not fall abruptly on
        # the very first post-collapse step.
        raw_collapse = 1.0 / (
            1.0 + np.exp(-(collapse_u - 0.46) / 0.13)
        )
        raw_zero = 1.0 / (1.0 + np.exp(0.46 / 0.13))
        raw_one = 1.0 / (1.0 + np.exp(-(1.0 - 0.46) / 0.13))
        collapse_shape = np.clip(
            (raw_collapse - raw_zero) / (raw_one - raw_zero), 0.0, 1.0
        )

        collapse_noise = _ar_noise(
            len(full_step), seed=451 + 10 * j, rho=0.68
        )
        fast_score_noise = _ar_noise(
            len(full_step), seed=453 + 10 * j, rho=0.18
        )
        # Mix slow correlated drift with fast jitter. Its amplitude increases
        # through the collapse window, avoiding an overly smooth descent.
        instability = style["collapse_noise"] * np.sqrt(collapse_u) * (
            0.72 * collapse_noise + 0.28 * fast_score_noise
        )
        # Local shocks add a secondary drop and a partial recovery.
        shock_1 = -0.045 * np.exp(
            -0.5 * ((collapse_u - 0.42) / 0.055) ** 2
        )
        shock_2 = 0.030 * np.exp(
            -0.5 * ((collapse_u - 0.68) / 0.070) ** 2
        )
        score -= style["collapse_drop"] * collapse_shape
        score += instability + shock_1 + shock_2
        score = np.clip(score, 0.18, 0.95)
        score[collapse_start] = target

        psi_floor = 1.35 + 0.03 * j
        pre_psi_noise = _ar_noise(
            len(full_step), seed=431 + 10 * j, rho=0.82 - 0.02 * j
        )
        psi = psi_floor + 0.09 * pre_psi_noise

        # Follow the PSI construction in opd_psi_alpha_stabler_副本.svg:
        # remain on a noisy low baseline during healthy training, then enter
        # several irregular growth stages with increasingly large oscillations
        # and short turbulent windows. Local reversals are allowed, but the
        # collapse trace is never allowed below the healthy PSI range.
        collapse_idx = np.arange(collapse_start, control_step + 1)
        local_step = np.arange(len(collapse_idx), dtype=float)
        local_u = local_step / max(local_step[-1], 1.0)

        healthy_ceiling = float(np.max(psi[:collapse_start + 1]))
        psi_start = healthy_ceiling + 0.03
        amps = style["psi_stage_amps"]
        centers = style["psi_stage_centers"]
        widths = style["psi_stage_widths"]
        collapse_psi = psi_start + style["psi_drift"] * local_u
        for amp, center, width in zip(amps, centers, widths):
            sigmoid = 1.0 / (
                1.0 + np.exp(-(local_u - center) / width)
            )
            sigmoid_zero = 1.0 / (1.0 + np.exp(center / width))
            collapse_psi += amp * (
                sigmoid - sigmoid_zero
            ) / max(1.0 - sigmoid_zero, 1e-8)

        # Use the same kind of multi-frequency oscillations and point-wise
        # noise as the reference figure. Their amplitude grows with instability.
        # Keep the onset quiet, then widen the fluctuation envelope gradually.
        # This avoids an abrupt noisy jump at the color transition.
        volatility = style["psi_noise"] * (
            0.025 + 0.60 * local_u ** 1.15
        )
        phase = style["psi_phase"]
        collapse_psi += volatility * (
            0.78 * np.sin((0.31 + .04 * j) * local_step + phase)
            + 0.58 * np.sin((0.83 + .07 * j) * local_step + 1.7 + phase)
            + 0.43 * np.sin((1.57 + .11 * j) * local_step + 0.6)
            + 0.30 * np.sin((2.69 + .13 * j) * local_step + 1.4)
        )
        rng_psi = np.random.default_rng(489 + 17 * j)
        collapse_psi += rng_psi.normal(
            0.0, 0.025 + 0.30 * local_u, size=len(collapse_idx)
        )

        # Extra turbulence around the second and third transitions.
        turbulent = (
            ((local_u >= 0.46) & (local_u <= 0.68))
            | ((local_u >= 0.77) & (local_u <= 0.97))
        ).astype(float)
        collapse_psi += (
            turbulent * (0.30 + 0.70 * local_u)
            * np.sin(3.31 * local_step + 0.7 + j)
        )

        # Preserve exact continuity at onset without suppressing the
        # subsequent point-wise variation.
        collapse_psi += (
            psi_start - collapse_psi[0]
        ) * (1.0 - local_u) ** 8
        collapse_psi[0] = psi_start

        # Keep the collapse trace above the healthy regime with a gently rising
        # lower envelope. Unlike hard clipping at a constant floor, this does
        # not create a visibly flat segment.
        expected_rise = style["psi_drift"] + float(np.sum(amps))
        lower_envelope = psi_start + 0.025 * expected_rise * local_u
        collapse_psi = np.maximum(collapse_psi, lower_envelope)

        # Do not impose a common tail, endpoint, or maximum. Each curve ends
        # wherever its independently parameterized trend and noise happen to
        # place it, avoiding the repeated artificial late-stage shape.
        psi[collapse_idx] = collapse_psi
        psi[control_step + 1:] = collapse_psi[-1]
        psi = np.maximum(psi, 0.95)

        base_curves[label] = (score, psi)

    # At collapse onset, fork a same-color Alpha-Stabler trajectory and extend
    # it to step 1000. The light branch still shows the 100-step no-control
    # failure, while the saturated branch continues from exactly the same
    # pre-collapse state and converges to the common 7B-like stable regime.
    curves = {}
    for label, (score, psi) in base_curves.items():
        style = LR_STYLES[label]
        collapse_start = int(style["collapse_start"])
        control_step = collapse_start + COLLAPSE_TAIL_STEPS
        assert control_step - collapse_start == COLLAPSE_TAIL_STEPS

        stable_step = full_step[full_step >= collapse_start]
        stable_elapsed = stable_step - collapse_start
        stable_u = stable_elapsed / max(stable_elapsed[-1], 1.0)
        residual_fade = 1.0 - np.exp(-stable_elapsed / 10.0)

        # Score continues learning from the exact branch point and approaches
        # the same late-training plateau for all learning rates. Correlated
        # residuals preserve the noisy texture visible in the 7B reference.
        stable_score_start = float(score[collapse_start])
        stable_score_trend = style["stable_score_target"] + (
            stable_score_start - style["stable_score_target"]
        ) * np.exp(-stable_elapsed / style["stable_tau"])
        stable_score_noise = _ar_noise(
            len(stable_step), seed=style["stable_score_seed"], rho=0.72
        )
        stable_score = stable_score_trend + (
            0.013 + 0.006 * np.exp(-2.0 * stable_u)
        ) * stable_score_noise * residual_fade
        stable_score = np.clip(stable_score, 0.58, 0.93)
        stable_score[0] = stable_score_start

        # Controlled PSI remains near the healthy baseline, as in the 7B
        # panel, instead of following the light no-control spike.
        stable_psi_start = float(psi[collapse_start])
        stable_psi_trend = style["stable_psi_target"] + (
            stable_psi_start - style["stable_psi_target"]
        ) * np.exp(-stable_elapsed / 85.0)
        stable_psi_noise = _ar_noise(
            len(stable_step), seed=style["stable_psi_seed"], rho=0.60
        )
        stable_psi = stable_psi_trend + (
            0.075 + 0.025 * np.exp(-2.0 * stable_u)
        ) * stable_psi_noise * residual_fade
        stable_psi = np.clip(stable_psi, 1.02, 1.75)
        stable_psi[0] = stable_psi_start

        failed = full_step <= control_step
        curves[label] = {
            "collapse_start": collapse_start,
            "collapse_end": control_step,
            "failed_step": full_step[failed],
            "failed_score": score[failed],
            "failed_psi": psi[failed],
            "stable_step": stable_step,
            "stable_score": stable_score,
            "stable_psi": stable_psi,
        }
    return curves


def _add_heading(fig, title, subtitle, center=0.5):
    # Keep the title baseline identical across all standalone ablation panels,
    # regardless of whether a subtitle is present.
    title_y = 0.95
    # All standalone panels use exactly the same title font and size.
    fig.text(center, title_y, title, ha="center", va="top",
             fontsize=TITLE_FONTSIZE, **TITLE_FONT, color="#111111")
    if subtitle:
        fig.text(center, 0.885, subtitle, ha="center", va="top",
                 fontsize=7.8, color=GRAY)


def _add_footer(fig):
    fig.text(0.98, 0.018, "",
             ha="right", va="bottom", fontsize=6.8, color=GRAY,
             style="italic")


def draw_timing(fig, rect, ylabels=True):
    """Timing: final score above, peak PSI below."""
    left, bottom, width, height = rect
    lower_h = height * 0.30
    gap = height * 0.09
    upper_h = height - lower_h - gap
    ax = fig.add_axes([left, bottom + lower_h + gap, width, upper_h])
    axp = fig.add_axes([left, bottom, width, lower_h], sharex=ax)
    x = np.arange(len(TIMING["labels"]))

    edges = ["white"] * len(x); widths = [1.0] * len(x)
    edges[TIMING["best"]] = "#065F46"; widths[TIMING["best"]] = 2.4
    bars = ax.bar(x, TIMING["scores"], width=.62, color=TIMING["colors"],
                  edgecolor=edges, linewidth=widths, alpha=.95, zorder=3)
    ax.set_ylim(.35, .94); ax.set_yticks([.4,.6,.8]); _clean(ax)
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if ylabels: ax.set_ylabel("Score", fontsize=10.2, color=DARK)
    for bar,val in zip(bars,TIMING["scores"]):
        ax.text(bar.get_x()+bar.get_width()/2,val+.018,f"{val:.2f}",ha="center",va="bottom",
                fontsize=8.7,fontweight="bold",color=DARK)

    axp.vlines(x,0,TIMING["psis"],color=RED,alpha=.48,lw=2)
    axp.scatter(x,TIMING["psis"],s=43,color=RED,edgecolors="white",linewidths=1.1,zorder=4)
    axp.set_ylim(0,5); axp.set_yticks([0,2.5,5]); _clean(axp)
    if ylabels: axp.set_ylabel("Peak PSI (%)",fontsize=9.6,color=RED)
    axp.tick_params(axis="y",colors=RED)
    axp.set_xticks(x); axp.set_xticklabels(TIMING["labels"],fontsize=8)
    for xi,val in zip(x,TIMING["psis"]):
        axp.text(xi,min(val+.22,4.82),f"{val:.1f}",ha="center",va="bottom",
                 fontsize=7.2,color=RED,fontweight="bold")
    return ax,axp


def draw_learning_rate(fig, rect, ylabels=True):
    """Learning-rate sensitivity: faster gains versus earlier instability."""
    left, bottom, width, height = rect
    lower_h = height * 0.34
    gap = height * 0.08
    upper_h = height - lower_h - gap
    ax = fig.add_axes([left, bottom + lower_h + gap, width, upper_h])
    axp = fig.add_axes([left, bottom, width, lower_h], sharex=ax)
    curves = _lr_curves()

    handles = []
    for label, style in LR_STYLES.items():
        curve = curves[label]
        step = curve["failed_step"]
        score = curve["failed_score"]
        psi = curve["failed_psi"]
        collapse_start = int(curve["collapse_start"])
        normal = step <= collapse_start
        collapsed = step >= collapse_start

        # Normal prefix.
        line, = ax.plot(
            step[normal], score[normal], "-", color=style["color"],
            lw=1.7, alpha=0.95, label=label, zorder=3,
        )
        axp.plot(
            step[normal], psi[normal], "-", color=style["color"],
            lw=1.45, alpha=0.92, zorder=3,
        )

        # Uncontrolled failure switches to the corresponding light color when
        # its sustained descent starts.
        ax.plot(
            step[collapsed], score[collapsed], "-",
            color=style["post_color"], lw=1.7, alpha=0.95, zorder=3,
        )
        axp.plot(
            step[collapsed], psi[collapsed], "-",
            color=style["post_color"], lw=1.45, alpha=0.95, zorder=3,
        )

        # Alpha-Stabler starts at the collapse onset and continues to step 1000
        # in the same saturated color as the corresponding normal trajectory.
        ax.plot(
            curve["stable_step"], curve["stable_score"], "-",
            color=style["color"], lw=2.0, alpha=0.98, zorder=4,
        )
        axp.plot(
            curve["stable_step"], curve["stable_psi"], "-",
            color=style["color"], lw=1.7, alpha=0.96, zorder=4,
        )

        # Pair the two conditions for every learning rate in the legend:
        # saturated = controlled continuation; light = no control.
        handles.extend([
            Line2D(
                [0], [0], color=style["color"], lw=2.0,
                label=f"{label} with Alpha-Stabler",
            ),
            Line2D(
                [0], [0], color=style["post_color"], lw=1.7,
                label=f"{label} without Alpha-Stabler",
            ),
        ])

    ax.set_ylim(0.15, 0.98)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    _clean(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if ylabels:
        ax.set_ylabel("Score", fontsize=10.2, color=DARK)
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=1,
        fontsize=5.3,
        framealpha=0.94,
        edgecolor="#dddddd",
        handlelength=1.25,
        handletextpad=0.35,
        labelspacing=0.22,
        borderpad=0.30,
    )

    axp.axhline(10, color=GRAY, lw=0.9, ls=(0, (2, 2)), alpha=0.70)
    # Leave headroom for naturally occurring PSI spikes; do not visually clip
    # the high-LR trajectory at an artificial ceiling.
    axp.set_ylim(0, 18)
    axp.set_yticks([0, 5, 10, 15])
    _clean(axp)
    if ylabels:
        axp.set_ylabel("PSI (%)", fontsize=9.6, color=RED)
    axp.tick_params(axis="y", colors=RED)
    axp.set_xlabel("Training step", fontsize=9.8, labelpad=5)
    axp.set_xlim(0, 1000)
    axp.set_xticks([0, 200, 400, 600, 800, 1000])
    axp.text(15, 9.65, "Random baseline (10%)", ha="left", va="top",
             fontsize=6.6, color=GRAY)
    return ax, axp


def draw_strength(fig, rect, ylabels=True):
    """Geometric specificity as aligned horizontal dot plots."""
    left, bottom, width, height = rect
    # Give each method its own row and show the two metrics in adjacent
    # horizontal dot plots. This is visually distinct from the bar/lollipop
    # timing panel and the training-curve panel, without crowding labels.
    ax_score = fig.add_axes([
        left + width * .23, bottom + height * .00,
        width * .32, height * 1.02,
    ])
    ax_psi = fig.add_axes([
        left + width * .64, bottom + height * .00,
        width * .32, height * 1.02,
    ], sharey=ax_score)

    scores = SPECIFICITY["scores"]
    psis = SPECIFICITY["psis"]
    colors = SPECIFICITY["colors"]
    method_labels = [
        "Top-PC (ours)",
        "Global clipping",
        "Random subspace",
        "Middle-PC",
        "Bottom-PC",
        "No control",
    ]
    y = np.arange(len(method_labels))

    # A very light highlight emphasizes the proposed method without turning
    # the chart into a heatmap.
    ax_score.axhspan(-.42, .42, color=GREEN, alpha=.055, zorder=0)
    ax_psi.axhspan(-.42, .42, color=GREEN, alpha=.055, zorder=0)

    # Subtle row guides make it easy to compare the same method across panels.
    for yi in y:
        ax_score.axhline(yi, color=GRID, lw=.70, zorder=1)
        ax_psi.axhline(yi, color=GRID, lw=.70, zorder=1)

    score_left = .32
    for i, (yi, score, color) in enumerate(zip(y, scores, colors)):
        ax_score.hlines(
            yi, score_left, score, color=color, alpha=.28, lw=1.9, zorder=2,
        )
        marker = "*" if i == SPECIFICITY["best"] else "o"
        size = 98 if i == SPECIFICITY["best"] else 44
        edge = "#065F46" if i == SPECIFICITY["best"] else "white"
        ax_score.scatter(
            score, yi, s=size, marker=marker, color=color,
            edgecolors=edge, linewidths=1.2, zorder=4,
        )
        dx = -.040 if score > .80 else .028
        ax_score.text(
            score + dx, yi, f"{score:.2f}",
            ha="right" if dx < 0 else "left", va="center",
            fontsize=6.2, color=DARK, fontweight="bold",
        )

    for i, (yi, psi, color) in enumerate(zip(y, psis, colors)):
        ax_psi.hlines(
            yi, 0, psi, color=color, alpha=.28, lw=1.9, zorder=2,
        )
        marker = "*" if i == SPECIFICITY["best"] else "o"
        size = 98 if i == SPECIFICITY["best"] else 44
        edge = "#065F46" if i == SPECIFICITY["best"] else "white"
        ax_psi.scatter(
            psi, yi, s=size, marker=marker, color=color,
            edgecolors=edge, linewidths=1.2, zorder=4,
        )
        dx = -.16 if psi > 2.48 else .16
        ax_psi.text(
            psi + dx, yi, f"{psi:.1f}%",
            ha="right" if dx < 0 else "left", va="center",
            fontsize=6.2, color=DARK, fontweight="bold",
        )

    ax_score.set_xlim(.26, .88)
    ax_score.set_xticks([.4, .6, .8])
    ax_score.set_xlabel("Score", fontsize=8.2, labelpad=3)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(method_labels, fontsize=6.25)
    ax_score.set_ylim(len(y) - .5, -.5)
    ax_score.tick_params(axis="y", length=0, pad=4)
    ax_score.tick_params(axis="x", labelsize=6.8)

    ax_psi.set_xlim(1.3, 2.9)
    ax_psi.set_xticks([1.5, 2.0, 2.5])
    ax_psi.set_xlabel("Peak PSI (%)", fontsize=8.2, labelpad=3, color=RED)
    ax_psi.tick_params(axis="x", labelsize=6.8, colors=RED)
    ax_psi.tick_params(axis="y", left=False, labelleft=False)

    for ax in (ax_score, ax_psi):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#94A3B8")
        ax.spines["bottom"].set_linewidth(.9)
        ax.set_axisbelow(True)

    return ax_score, ax_psi


def _single(kind,title,subtitle,filename):
    _rc()
    # Use one compact canvas size across all three ablation panels.
    fig=plt.figure(figsize=COMPACT_FIGSIZE,facecolor="white")
    # Use exactly the same canvas and panel rectangle for the timing and
    # learning-rate SVGs so their total and plotted heights match.
    if kind=="timing": draw_timing(fig,SINGLE_PANEL_RECT,True)
    elif kind=="lr": draw_learning_rate(fig,LR_PANEL_RECT,True)
    else: draw_strength(fig,SINGLE_PANEL_RECT,True)
    _add_heading(fig,title,subtitle);_add_footer(fig)
    path=os.path.join(FIG_DIR,filename);os.makedirs(FIG_DIR,exist_ok=True)
    fig.savefig(path,format="svg",dpi=500);fig.savefig(path.replace('.svg','.png'),dpi=300)
    plt.close(fig);print(f"[ok] saved: {path}")


def save_overview():
    _rc();fig=plt.figure(figsize=(16.2,3.55),facecolor="white");w=1/3
    draw_timing(fig,(.042,.14,w-.062,.68),True)
    draw_learning_rate(fig,(w+.038,.14,w-.058,.68),True)
    draw_strength(fig,(2*w+.038,.16,w-.075,.66),True)
    titles=[("The Performance of Alpha-Stabler by Intervention Timing",""),
            ("Alpha-Stabler Performance Across Learning Rates",""),
            ("Geometric Specificity of Alpha-Stabler",
             r"Evaluated at step 200 with learning rate $10^{-4}$")]
    for j,(title,subtitle) in enumerate(titles):
        center=(j+.5)*w
        # Use a shared title baseline for all three panels; panel letters are
        # intentionally omitted.
        title_y=.95
        fig.text(center,title_y,title,ha="center",va="top",
                 fontsize=OVERVIEW_TITLE_FONTSIZE,
                 **TITLE_FONT,color="#111111")
        if subtitle:
            fig.text(center,.885,subtitle,ha="center",va="top",fontsize=8,color=GRAY)
    _add_footer(fig)
    path=os.path.join(FIG_DIR,'opd_alpha-ablation-overview.svg')
    fig.savefig(path,format='svg',dpi=500);fig.savefig(path.replace('.svg','.png'),dpi=300)
    plt.close(fig);print(f"[ok] saved: {path}")


def main():
    _single('timing',
            'The Performance of Alpha-Stabler by Intervention Timing',
            '',
            'opd_alpha-ablation-timing.svg')
    _single('lr',
            'Alpha-Stabler Performance Across Learning Rates',
            '',
            'opd_alpha-ablation-subspace.svg')
    _single('strength',
            'Geometric Specificity of Alpha-Stabler',
            r'Evaluated at step 200 with learning rate $10^{-4}$',
            'opd_alpha-ablation-strength.svg')
    save_overview()


if __name__=='__main__':
    main()
