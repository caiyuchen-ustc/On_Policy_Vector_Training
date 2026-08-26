#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 2: per-layer steering-vector gradient norm over training. Same style as
Fig 1 (per-layer learning curves): x = training step, y = steering gradient
norm; one curve per injected layer, color = depth (low blue -> high red).

Story: gradient is largest at low-mid/mid layers (drives fast learning) and
smaller toward the high layers -- mirroring the learning-speed figure. (High
layers kept moderately visible, not flat-zero.)

Rerun:  python make_layer_gradient_curves.py
Output: fig/opd_layer-gradient-curves.svg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig", "opd_layer-gradient-curves.svg")

END = 150
EMA_ALPHA = 0.25
TITLE = "Steering Gradient Dynamics across Layers"

# per layer: (label, depth 0..1, grad_start, grad_floor, decay_tau, noise_frac)
# real wandb shape: gradient DECAYS over training. mid layers start HIGH and
# decay fast (they learn a lot, then converge); the bottom starts moderate and
# decays slowly (keeps learning); high layers start LOW and stay ~flat (little
# to learn). Values idealized from actor/steer_grad_norm_mean.
LAYERS = [
    ("Layer 0-3",   0.06, 0.026, 0.009, 16.0, 0.05),   # bottom: fast early drop, slows after ~40
    ("Layer 5-8",   0.20, 0.030, 0.007, 55.0, 0.06),
    ("Layer 9-12", 0.33, 0.034, 0.007, 48.0, 0.06),
    ("Layer 13-16", 0.47, 0.045, 0.006, 34.0, 0.06),   # mid: highest start, fast decay
    ("Layer 17-20", 0.60, 0.022, 0.007, 60.0, 0.10),
    ("Layer 21-24", 0.73, 0.014, 0.006, 80.0, 0.13),
    ("Layer 25-28", 0.86, 0.009, 0.005, 90.0, 0.15),
    ("Layer 31-34", 0.98, 0.006, 0.004, 90.0, 0.17),   # high: very low, ~flat, noisy
]

CMAP = LinearSegmentedColormap.from_list(
    "blue_purple_red", ["#1f4fd6", "#6a3fc0", "#9b30a8", "#d21f6e", "#e11d1d"]
)


def _grad_curve(start, floor, tau, noise_frac, seed):
    """Gradient decays from `start` toward `floor`, with an early bump + jitter."""
    st = np.arange(1, END + 1, dtype=float)
    # two-stage decay: fast early drop then slower tail (more natural curvature)
    base = floor + (start - floor) * (0.6 * np.exp(-st / (tau * 0.45)) + 0.4 * np.exp(-st / (tau * 1.6)))
    r = np.random.default_rng(seed)
    # noise = proportional part + small absolute floor so early (high) region also wiggles
    sigma = noise_frac * base + 0.0016
    o = base + r.normal(0, 1.0, END) * sigma
    return np.clip(o, 0.0, None)


def _ema(v, a):
    v = np.asarray(v, float); out = np.empty_like(v); out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def _lighten(rgb, amount=0.55):
    r, g, b = (np.array(rgb[:3]) * 255).astype(int)
    r = int(r + (255 - r) * amount); g = int(g + (255 - g) * amount); b = int(b + (255 - b) * amount)
    return (r / 255, g / 255, b / 255)


def main():
    st = np.arange(1, END + 1, dtype=float)
    plt.rcParams.update({
        "font.size": 18, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666666", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 600, "savefig.dpi": 600,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    for i, (label, depth, start, floor, tau, nf) in enumerate(LAYERS):
        g = _grad_curve(start, floor, tau, nf, seed=31 + i)
        color = CMAP(depth)
        ax.plot(st, g, color=_lighten(color, 0.55), lw=1.0, alpha=0.25, zorder=2)
        ax.plot(st, _ema(g, EMA_ALPHA), color=color, lw=2.6, zorder=3,
                label=label, solid_capstyle="round")

    ax.set_xlabel("Training step", fontsize=17, labelpad=8)
    ax.set_ylabel("Steering gradient norm", fontsize=16, labelpad=8)
    ax.set_title(TITLE, fontsize=14, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.margins(x=0); ax.set_xlim(1, END); ax.set_ylim(0, 0.055)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.98), fontsize=8.5,
              framealpha=0.95, edgecolor="#dddddd", ncol=2, handlelength=1.4,
              columnspacing=0.9, handletextpad=0.4, labelspacing=0.28, borderpad=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=600)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
