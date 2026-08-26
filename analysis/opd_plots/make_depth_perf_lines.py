#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-depth performance line plot: x = injected/trained depth (bottom->high),
y = converged performance. One line per method.

Story: Vector steering (a single additive vector, no input-dependent
non-linearity) must borrow the downstream non-linear blocks, so it collapses
when injected at HIGH layers. LoRA (any rank) and Full-param carry their own
input-dependent capacity and stay high at every depth. => RL can be contained
in very few parameters, but only above a minimum non-linear capacity; not *any*
few / *any* position.

Edit CONFIG and rerun:
    python make_depth_perf_lines.py
Output: fig/opd_depth-perf-lines.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_depth-perf-lines.svg")

TITLE = "Where RL Can Be Contained: Performance vs. Depth"
X_LABELS = ["Bottom", "Low", "Low-Mid", "Mid", "Mid-High", "High", "Top"]  # 7 depth bins

# Vector Steering: full 7-point sweep (real). LoRA & Full are ~flat and high;
# LoRA and Full are basically the same, so we DON'T draw 5 overlapping lines.
# Instead: one Vector line, one Full-param line, and LoRA drawn as a shaded
# band (min..max across ranks) + a single mean line -> clean, uncluttered.
#   >>> replace numbers with your real sweep <<<
X_LABELS = ["Bottom", "Low", "Low-Mid", "Mid", "Mid-High", "High", "Top"]  # 7 depth bins

VECTOR = [0.86, 0.86, 0.84, 0.80, 0.66, 0.42, 0.22]
FULL   = [0.90, 0.90, 0.89, 0.90, 0.89, 0.90, 0.89]
# LoRA ranks (each ~flat); band = min..max, line = mean.
LORA_RANKS = {
    "LoRA-4":  [0.84, 0.84, 0.85, 0.84, 0.85, 0.84, 0.83],
    "LoRA-8":  [0.85, 0.86, 0.85, 0.86, 0.85, 0.86, 0.85],
    "LoRA-16": [0.86, 0.87, 0.86, 0.87, 0.86, 0.87, 0.86],
    "LoRA-32": [0.87, 0.88, 0.87, 0.88, 0.87, 0.88, 0.87],
}

C_VECTOR = "#2ca02c"    # green
C_LORA   = "#d62728"    # red band
C_FULL   = "#1f77b4"    # blue

FAIL_THRESH = 0.5     # perf below this = "failure" (shade region under it)


def main():
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#e8e8ec", "grid.linewidth": 0.9,
        "xtick.labelsize": 12.5, "ytick.labelsize": 13,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    x = np.arange(len(X_LABELS))

    # faint fail-zone band at the bottom
    ax.axhspan(0.0, FAIL_THRESH, color="#d62728", alpha=0.05, zorder=0)
    ax.axhline(FAIL_THRESH, color="#d62728", ls=":", lw=1.3, alpha=0.5, zorder=1)
    ax.text(len(X_LABELS) - 0.75, FAIL_THRESH - 0.045, "Failure region",
            color="#d62728", fontsize=10.5, alpha=0.75, ha="right", va="top")

    # LoRA band (min..max across ranks) + mean line
    lora = np.array(list(LORA_RANKS.values()), float)
    lo, hi, mean = lora.min(0), lora.max(0), lora.mean(0)
    ax.fill_between(x, lo, hi, color=C_LORA, alpha=0.18, zorder=2, linewidth=0)
    ax.plot(x, mean, color=C_LORA, ls="--", lw=2.4, marker="s", markersize=6.5,
            markeredgecolor="white", markeredgewidth=1.1,
            label="LoRA (r=4–32)", zorder=4, solid_capstyle="round")

    # Full-param
    ax.plot(x, FULL, color=C_FULL, ls="-", lw=2.6, marker="D", markersize=7,
            markeredgecolor="white", markeredgewidth=1.2, label="Full-param",
            zorder=4, solid_capstyle="round")

    # Vector steering (the star of the plot)
    ax.plot(x, VECTOR, color=C_VECTOR, ls="-", lw=3.2, marker="o", markersize=9.5,
            markeredgecolor="white", markeredgewidth=1.3, label="Vector Steering",
            zorder=5, solid_capstyle="round")

    ax.set_xticks(x); ax.set_xticklabels(X_LABELS, rotation=16, ha="right")
    ax.set_xlim(-0.3, len(X_LABELS) - 0.7)
    ax.set_xlabel("Injected / trained depth", fontsize=17, labelpad=8)
    ax.set_ylabel("Converged score", fontsize=17, labelpad=8)
    ax.set_ylim(0.12, 0.96)
    ax.set_title(TITLE, fontsize=15, pad=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.legend(loc="lower left", fontsize=11, framealpha=0.92, edgecolor="#dddddd",
              handlelength=2.0, labelspacing=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
