#!/usr/bin/env python3
"""Time/Score plots using the Score formulas in the user-supplied ablation script.

Dependencies: numpy, matplotlib
Run: python plot_pre_collapse_score_seconds.py --output-dir figs_pre_collapse
Optional exact reuse of the original full script:
    python plot_pre_collapse_score_seconds.py --source path/to/original_ablation.py
The optional source must be your trusted local Python script exposing _lr_curves()
and LR_STYLES. Importing it executes its top-level code. Without --source, the
built-in generator below reproduces the supplied Score equations and RNG seeds;
unused PSI, timing-bar and specificity plotting code are omitted.

By default Score values and clocks are preserved. --illustrative-noise 0.010
adds SIMULATED zero-mean variation with unchanged original Score endpoints.
--with-overhead 0.04 assumes 4% MORE elapsed time for the with-method curve at
EVERY matching step, including calibration. It does not slow Score convergence
in step space or claim that runtime overhead was measured. Both options apply
only to the built-in illustration and are blocked with --source.
Modified plots are visibly labelled; JSON records original scores and assumptions.
Example: python plot_pre_collapse_overhead.py --illustrative-noise 0.010 --with-overhead 0.04
Source curves are illustrative, not logs.
The source sets stable_score_target=0.88 for all three LRs. That is a trend
asymptote, NOT the last observed point of these generated curves. With the
supplied seeds, the final values are 0.862866/0.877120/0.864768 for LR
1e-4/5e-5/1e-5, matching 0.86/0.88/0.86 after rounding. Preserve source values;
do not force exact endpoints, remove noise, or impose a ranking at every step.

Estimated time: step 1 = 240 s, step 1000 = 420 s, linear growth, cumulative sum.
The 1000-step horizon belongs to the original source, not the earlier 300-step demo.
The baseline uses the estimated clock; with-method time is multiplied by
1 + with_overhead when explicitly requested. No empirical overhead is inferred.
Failed curves stop at their original 175/250/400 steps and are NOT extrapolated.
Controlled curves prepend their common pre-branch prefix for a fair origin.
Default plot: both methods from step 0 up to the source collapse-onset boundary
(75/150/300 for LR 1e-4/5e-5/1e-5). No post-collapse points are displayed.
Use --full-training to show the previous full curves instead. The time model
always spans the original 1000 updates, then is sliced: it is NOT rescaled to
reach 420 seconds per step at each cutoff. Estimated with-method overhead,
if enabled, scales the entire clock rather than adding a constant x-offset.
With-method means the normal warm-up/triggered setup, NOT always-on from step 1.
The source shares the pre-trigger prefix; without --illustrative-noise both Score
curves overlap exactly. Simulated variation, if requested, is not run evidence.
Outputs: three-panel PDF/PNG/SVG, all-LR comparison PDF/PNG/SVG, source metadata JSON.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Preserve the source dictionary order: RNG seeds depend on index j.
LR_STYLES = {
    r"$10^{-5}$": {
        "color": "#10B981", "post_color": "#86EFAC",
        "collapse_start": 300, "collapse_score": 0.75, "collapse_drop": 0.25,
        "growth_tau": 220.0, "noise_scale": 0.011, "collapse_noise": 0.050,
        "stable_score_target": 0.88, "stable_tau": 185.0, "stable_score_seed": 561,
    },
    r"$5{\times}10^{-5}$": {
        "color": "#2563EB", "post_color": "#93C5FD",
        "collapse_start": 150, "collapse_score": 0.70, "collapse_drop": 0.35,
        "growth_tau": 110.0, "noise_scale": 0.016, "collapse_noise": 0.065,
        "stable_score_target": 0.88, "stable_tau": 135.0, "stable_score_seed": 571,
    },
    r"$10^{-4}$": {
        "color": "#C96A72", "post_color": "#E8ADB2",
        "collapse_start": 75, "collapse_score": 0.65, "collapse_drop": 0.42,
        "growth_tau": 55.0, "noise_scale": 0.021, "collapse_noise": 0.078,
        "stable_score_target": 0.88, "stable_tau": 95.0, "stable_score_seed": 581,
    },
}
COLLAPSE_TAIL_STEPS = 100


def _ar_noise(n, seed, rho):
    """Unchanged source correlated-noise construction."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, n)
    out = np.empty(n, dtype=float)
    out[0] = eps[0]
    scale = np.sqrt(max(1.0 - rho ** 2, 1e-8))
    for j in range(1, n):
        out[j] = rho * out[j - 1] + scale * eps[j]
    return out


def _lr_curves():
    """Source Score branch of _lr_curves, with independent PSI work omitted."""
    full_step = np.arange(0, 1001, 1.0)
    base_curves = {}
    for j, (label, style) in enumerate(LR_STYLES.items()):
        collapse_start = int(style["collapse_start"])
        target = float(style["collapse_score"])
        tau = float(style["growth_tau"])
        effective_step = np.minimum(full_step, collapse_start)
        growth_fraction = ((1.0 - np.exp(-effective_step / tau)) /
                           (1.0 - np.exp(-collapse_start / tau)))
        score = 0.40 + (target - 0.40) * growth_fraction
        score_noise = _ar_noise(len(full_step), seed=421 + 10 * j, rho=0.82 - 0.03 * j)
        start_fade = 1.0 - np.exp(-full_step / 16.0)
        anchor_fade = np.clip(np.abs(full_step - collapse_start) / 12.0, 0.0, 1.0)
        score += style["noise_scale"] * score_noise * start_fade * anchor_fade
        collapse_u = np.clip((full_step - collapse_start) / COLLAPSE_TAIL_STEPS, 0.0, 1.0)
        raw_collapse = 1.0 / (1.0 + np.exp(-(collapse_u - 0.46) / 0.13))
        raw_zero = 1.0 / (1.0 + np.exp(0.46 / 0.13))
        raw_one = 1.0 / (1.0 + np.exp(-(1.0 - 0.46) / 0.13))
        collapse_shape = np.clip((raw_collapse - raw_zero) / (raw_one - raw_zero), 0.0, 1.0)
        collapse_noise = _ar_noise(len(full_step), seed=451 + 10 * j, rho=0.68)
        fast_score_noise = _ar_noise(len(full_step), seed=453 + 10 * j, rho=0.18)
        instability = style["collapse_noise"] * np.sqrt(collapse_u) * (
            0.72 * collapse_noise + 0.28 * fast_score_noise)
        shock_1 = -0.045 * np.exp(-0.5 * ((collapse_u - 0.42) / 0.055) ** 2)
        shock_2 = 0.030 * np.exp(-0.5 * ((collapse_u - 0.68) / 0.070) ** 2)
        score -= style["collapse_drop"] * collapse_shape
        score += instability + shock_1 + shock_2
        score = np.clip(score, 0.18, 0.95)
        score[collapse_start] = target
        base_curves[label] = score

    curves = {}
    for label, score in base_curves.items():
        style = LR_STYLES[label]
        collapse_start = int(style["collapse_start"])
        control_step = collapse_start + COLLAPSE_TAIL_STEPS
        stable_step = full_step[full_step >= collapse_start]
        stable_elapsed = stable_step - collapse_start
        stable_u = stable_elapsed / max(stable_elapsed[-1], 1.0)
        residual_fade = 1.0 - np.exp(-stable_elapsed / 10.0)
        stable_score_start = float(score[collapse_start])
        stable_score_trend = style["stable_score_target"] + (
            stable_score_start - style["stable_score_target"]
        ) * np.exp(-stable_elapsed / style["stable_tau"])
        stable_score_noise = _ar_noise(len(stable_step), seed=style["stable_score_seed"], rho=0.72)
        stable_score = stable_score_trend + (
            0.013 + 0.006 * np.exp(-2.0 * stable_u)
        ) * stable_score_noise * residual_fade
        stable_score = np.clip(stable_score, 0.58, 0.93)
        stable_score[0] = stable_score_start
        failed = full_step <= control_step
        curves[label] = {
            "collapse_start": collapse_start, "collapse_end": control_step,
            "failed_step": full_step[failed], "failed_score": score[failed],
            "stable_step": stable_step, "stable_score": stable_score,
        }
    return curves


def load_source(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("trusted_original_ablation", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the supplied Python source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._lr_curves(), module.LR_STYLES


def time_axis(total_steps, start_seconds=240.0, end_seconds=420.0):
    if total_steps < 2 or min(start_seconds, end_seconds) <= 0:
        raise ValueError("Need >=2 total steps and positive time endpoints.")
    if not np.isfinite([start_seconds, end_seconds]).all():
        raise ValueError("Time endpoints must be finite.")
    return np.r_[0.0, np.cumsum(np.linspace(start_seconds, end_seconds, total_steps))]


def assemble_score(curve):
    """Join existing prefix and existing controlled branch; change no Score value."""
    failed_steps = np.asarray(curve["failed_step"])
    failed_score = np.asarray(curve["failed_score"])
    stable_steps = np.asarray(curve["stable_step"])
    stable_score = np.asarray(curve["stable_score"])
    prefix = failed_steps < stable_steps[0]
    steps = np.r_[failed_steps[prefix], stable_steps]
    scores = np.r_[failed_score[prefix], stable_score]
    for x, y in [(steps, scores), (failed_steps, failed_score)]:
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Invalid source Score array lengths.")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
            raise ValueError("Source steps must increase and all values must be finite.")
        if np.any(x < 0) or not np.all(x == np.rint(x)):
            raise ValueError("This converter requires nonnegative integer source steps.")
    return steps.astype(int), scores, failed_steps.astype(int), failed_score


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#666666")
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.grid(True, color="#E2E8F0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Estimated wall-clock time (seconds)", fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax.set_ylim(0.15, 0.98)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.tick_params(labelsize=8)


def export_figure(fig, prefix):
    paths = []
    for suffix in [".png", ".pdf", ".svg"]:
        p = prefix.parent / (prefix.name + suffix)
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.06)
        paths.append(p)
    plt.close(fig)
    return paths


def illustrative_variation(n, amplitude=0.006, seed=711):
    """SIMULATED visual variation, never a substitute for measured run differences.

    Zero sample mean, exact zero endpoints, RMS <= amplitude, peak <= 2.5*amplitude.
    This is a deliberate modification to source-generated illustrative scores.
    """
    if n < 3 or not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Variation needs >=3 points and a finite nonnegative amplitude.")
    if amplitude == 0:
        return np.zeros(n)
    noise = _ar_noise(n, seed, rho=0.64)
    window = np.sin(np.linspace(0, np.pi, n))
    window[[0, -1]] = 0.0
    delta = window * (noise - np.sum(window * noise) / np.sum(window))
    rms = np.sqrt(np.mean(delta ** 2))
    if rms == 0:
        return np.zeros(n)
    delta *= min(amplitude / rms, 2.5 * amplitude / np.max(np.abs(delta)))
    return delta


def plot_source(curves, styles, output_dir, start_seconds=240.0, end_seconds=420.0,
                pre_collapse_only=True, illustrative_noise=0.0, with_overhead=0.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(reversed(list(curves)))
    if len(labels) != 3:
        raise ValueError("Expected three learning-rate curves.")
    original = {label: assemble_score(curves[label]) for label in labels}
    total_steps = max(int(v[0][-1]) for v in original.values())
    # Build the full training clock BEFORE slicing the visible window.
    clock = time_axis(total_steps, start_seconds, end_seconds)
    if not np.isfinite(with_overhead) or not 0 <= with_overhead <= 0.5:
        raise ValueError("Assumed overhead must be a finite fraction from 0 to 0.5.")
    with_clock = clock * (1.0 + with_overhead)
    assembled = {}
    source_scores = {}
    if not np.isfinite(illustrative_noise) or illustrative_noise < 0:
        raise ValueError("Illustrative noise must be finite and nonnegative.")
    for index, (label, (steps, scores, fs, fy)) in enumerate(original.items()):
        if pre_collapse_only:
            cutoff = int(curves[label]["collapse_start"])
            keep, keep_failed = steps <= cutoff, fs <= cutoff
            steps, scores, fs, fy = steps[keep], scores[keep], fs[keep_failed], fy[keep_failed]
        source_scores[label] = scores.copy()
        if illustrative_noise:
            scores = scores + illustrative_variation(len(scores), illustrative_noise, 711 + 17 * index)
        assembled[label] = (steps, scores, fs, fy)
    tag = "pre_collapse" if pre_collapse_only else "from_start"
    if illustrative_noise:
        tag += "_simulated_variation"
    if with_overhead:
        tag += f"_overhead{with_overhead * 100:g}pct"
    all_scores = np.concatenate([v[i] for v in assembled.values() for i in (1, 3)])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    pad = max(0.025, 0.08 * (hi - lo))
    note = ("ILLUSTRATIVE SOURCE DATA; ESTIMATED RUNTIME. "
            f"Original {total_steps}-step clock: {start_seconds:g}->{end_seconds:g} seconds per update.")
    footnote = (note + f"\nAssumed with-method elapsed-time overhead: {with_overhead:.0%}; "
                "source prefixes are illustrative, not measured runs.")
    if illustrative_noise:
        footnote = ("SIMULATED SCORE VARIATION + ASSUMED RUNTIME - NOT EXPERIMENTAL RESULTS.\n"
                    f"With-method: zero-mean illustrative variation (RMS <= {illustrative_noise:g}); "
                    f"assumed elapsed time +{with_overhead:.0%}; Score endpoints unchanged.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.75), sharex=False, sharey=True)
    for j, (ax, label) in enumerate(zip(axes, labels)):
        steps, scores, fs, fy = assembled[label]
        assert steps[0] == 0 and fs[0] == 0
        ax.axvspan(0, with_clock[min(50, int(steps[-1]))], color="#E5E7EB", alpha=0.45,
                   linewidth=0, zorder=0)
        ax.plot(with_clock[steps], scores, color="#0F9D76", linewidth=1.45,
                marker="o", markersize=2.4, markevery=max(1, len(steps) // 12),
                label="With Alpha-Stabler", zorder=3)
        ax.plot(clock[fs], fy, color="#D68138", linestyle=(0, (5, 3)), linewidth=1.15,
                marker="s", markersize=2.3, markevery=(max(1, len(fs)//24), max(1, len(fs)//12)),
                label="Without Alpha-Stabler", zorder=4)
        style_axes(ax)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(f"({chr(97+j)}) Learning rate {label}", fontsize=10, pad=9)
        ax.set_xlim(0, max(with_clock[steps[-1]], clock[fs[-1]]))
        ax.tick_params(labelleft=True)
        text = f"Steps 0-{int(steps[-1])}" if pre_collapse_only else "Full source window"
        ax.text(0.97, 0.055, text, transform=ax.transAxes,
                ha="right", fontsize=7.5, color="#666666")
    handles = [Line2D([], [], color="#0F9D76", lw=2.0, label="With Alpha-Stabler (warm-up + trigger)"),
               Line2D([], [], color="#D68138", lw=1.25, ls="--", label="Without Alpha-Stabler"),
               Patch(facecolor="#E5E7EB", alpha=0.45, label="With-method warm-up: 50 updates")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=8.2)
    fig.text(0.5, 0.012, footnote, ha="center", fontsize=6.6, color="#666666")
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.22, top=0.80, wspace=0.26)
    paths = export_figure(fig, output_dir / f"{tag}_time_score_panels")

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for label in labels:
        steps, scores, fs, fy = assembled[label]
        style = styles[label]
        ax.plot(with_clock[steps], scores, color=style["color"], linewidth=1.8,
                label=f"{label} with")
        ax.plot(clock[fs], fy, color=style["post_color"], linestyle="--", linewidth=1.0,
                label=f"{label} without")
    style_axes(ax)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlim(0, max(max(with_clock[v[0][-1]], clock[v[2][-1]]) for v in assembled.values()))
    ax.set_title("Pre-collapse Score versus time" if pre_collapse_only else "Score versus time", fontsize=11)
    ax.legend(loc="lower right", ncol=2, fontsize=8, framealpha=0.94)
    overlay_note = (f"ILLUSTRATIVE DATA; NOT EXPERIMENTAL RESULTS. Assumed with-method runtime +{with_overhead:.0%}."
                    + (" Added simulated Score variation." if illustrative_noise else ""))
    fig.text(0.5, 0.015, overlay_note, ha="center", fontsize=7, color="#666666")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.90)
    paths += export_figure(fig, output_dir / f"{tag}_time_score_overlay")

    summary = {"synthetic": True, "source": "user-supplied illustrative curve generator",
               "pre_collapse_only": pre_collapse_only,
               "simulated_score_variation": {"enabled": bool(illustrative_noise),
                                               "rms_limit": illustrative_noise,
                                               "peak_limit": 2.5 * illustrative_noise,
                                               "modified_method": "control",
                                               "not_measured_run_variation": True},
               "cutoff_rule": "Include collapse-onset boundary; exclude all later points.",
               "prefix_provenance": "Shared source prefix; not independent measured runs.",
               "time_comparison": "Assumed with-method time multiplier; not measured overhead.",
               "assumed_with_overhead_fraction": with_overhead,
               "assumed_with_time_multiplier": 1.0 + with_overhead,
               "warmup_updates": 50,
               "timing_model": {"first_step_seconds": start_seconds,
                                "last_step_seconds": end_seconds,
                                "total_steps": total_steps, "baseline_total_seconds": float(clock[-1]),
                                "with_total_seconds": float(with_clock[-1])},
               "curves": []}
    for label in labels:
        steps, scores, fs, fy = assembled[label]
        summary["curves"].append({
            "learning_rate": label,
            "stable_target": styles[label]["stable_score_target"],
            "actual_final_source_score": float(original[label][1][-1]),
            "last_visible_score": float(scores[-1]),
            "last_visible_seconds": float(with_clock[steps[-1]]),
            "baseline_last_visible_seconds": float(clock[fs[-1]]),
            "collapse_start_step": int(curves[label]["collapse_start"]),
            "control_source_score_before_variation": source_scores[label].tolist(),
            "control": {"steps": steps.tolist(), "time_seconds": with_clock[steps].tolist(), "score": scores.tolist()},
            "baseline": {"steps": fs.tolist(), "time_seconds": clock[fs].tolist(), "score": fy.tolist()}})
    p = output_dir / f"{tag}_time_score.synthetic.json"
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.append(p)
    return paths, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="Your trusted original .py exposing _lr_curves()")
    parser.add_argument("--output-dir", type=Path, default=Path("figs_pre_collapse"))
    parser.add_argument("--full-training", action="store_true", help="Show full source curves instead of pre-collapse only")
    parser.add_argument("--start-step-seconds", type=float, default=240)
    parser.add_argument("--end-step-seconds", type=float, default=420)
    parser.add_argument("--illustrative-noise", type=float, default=0.0,
                        help="Explicit SIMULATED variation amplitude (e.g. 0.010); default 0 preserves source")
    parser.add_argument("--with-overhead", type=float, default=0.0,
                        help="ASSUMED extra elapsed-time fraction for with-method (e.g. 0.04 = 4%%)")
    args = parser.parse_args()
    if args.source and (args.illustrative_noise or args.with_overhead):
        parser.error("Simulated variation/overhead apply only to the built-in illustration, not external data.")
    curves, styles = load_source(args.source) if args.source else (_lr_curves(), LR_STYLES)
    paths, summary = plot_source(curves, styles, args.output_dir,
                                 args.start_step_seconds, args.end_step_seconds,
                                 pre_collapse_only=not args.full_training,
                                 illustrative_noise=args.illustrative_noise,
                                 with_overhead=args.with_overhead)
    for row in summary["curves"]:
        print(row["learning_rate"], "last shown step:", row["control"]["steps"][-1],
              "score:", round(row["last_visible_score"], 6),
              "estimated seconds:", round(row["last_visible_seconds"], 3))
    print("Underlying full-run clock remains", summary["timing_model"]["total_steps"], "steps.")
    for p in paths:
        print(p.resolve())


if __name__ == "__main__":
    main()