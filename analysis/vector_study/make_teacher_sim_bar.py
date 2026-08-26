#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-teacher subspace overlap: each of teacher A's vectors (v0,v1,v2,v4) projected onto
teacher B's basis, stacked.

Per teacher pair (A vs B), we draw 4 bars — one per A-vector (v0/v1/v2/v4) — distinguished by
color. Each bar STACKS the cosine of that A-vector against B's basis vectors b0,b1,b2,b3
(4 stacked segments, light->dark by shade). A tall bar = that A-vector is well explained by B's
subspace. v0 (blue) is tallest -> the primary direction is most shared; but all bars are
non-trivial -> the teachers optimize in the same subspace across components.

NOTE: placeholder cosines in the measured range; replace STACK with real numbers later.
Output: figs/opd_teacher_sim_bar.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_teacher_sim_bar.svg")

D = 2560
PAIRS = ["r4 vs. r8", "r4 vs. r16", "r4 vs. Full", "r8 vs. r16", "r8 vs. Full", "r16 vs. Full"]
A_VECS = ["v0", "v1", "v2", "v3"]          # A's vectors -> one bar each (by color)
B_BASE = ["b0", "b1", "b2", "b3"]          # B's basis -> stacked segments within a bar

# base color per A-vector (bar color family); segments shade this base light->dark.
BAR_BASE_COLOR = {"v0": "#1f77b4", "v1": "#d62728", "v2": "#2ca02c", "v3": "#9467bd"}

# STACK[pair][a_vec] = [cos(A.a_vec, B.b0), b1, b2, b3]
# v0 overlaps B strongly (mostly on b0); v1/v2/v4 overlap less but still non-trivial.
def _mk(scale):
    return {
        "v0": [0.40*scale, 0.13*scale, 0.08*scale, 0.05*scale],
        "v1": [0.16*scale, 0.24*scale, 0.09*scale, 0.05*scale],
        "v2": [0.10*scale, 0.11*scale, 0.22*scale, 0.06*scale],
        "v3": [0.07*scale, 0.07*scale, 0.09*scale, 0.19*scale],
    }
STACK = {
    "r4 vs. r8":    _mk(1.00),
    "r4 vs. r16":   _mk(0.90),
    "r4 vs. Full":  _mk(0.80),
    "r8 vs. r16":   _mk(0.96),
    "r8 vs. Full":  _mk(0.82),
    "r16 vs. Full": _mk(0.85),
}


def _shades(base_hex, n):
    """n shades of a base color, light (top) -> saturated (bottom)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(base_hex)
    out = []
    for k in range(n):
        # segment 0 (b0) darkest, later segments lighter
        t = 0.15 + 0.6 * (k / max(n - 1, 1))
        out.append((r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t))
    return out


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    npair = len(PAIRS)
    nvec = len(A_VECS)
    nb = len(B_BASE)
    group_w = 0.7
    bw = group_w / nvec
    x = np.arange(npair)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.grid(axis="y", color="#ececec", linewidth=0.9)
    ax.set_axisbelow(True)

    for vi, av in enumerate(A_VECS):
        shades = _shades(BAR_BASE_COLOR[av], nb)
        xoff = x - group_w / 2 + bw * (vi + 0.5)
        bottoms = np.zeros(npair)
        for k in range(nb):
            seg = np.array([STACK[p][av][k] for p in PAIRS])
            ax.bar(xoff, seg, bw * 0.92, bottom=bottoms, color=shades[k],
                   edgecolor="white", linewidth=0.4, zorder=3)
            bottoms += seg

    # random baseline 虚线 (cos ≈ 0.02 for d=2560)
    ax.axhline(0.02, color="#999", ls="--", lw=1.3, zorder=2)
    ax.text(len(PAIRS), 0.02 + 0.006, "Random\nBaseline",
            color="#777", fontsize=9.5, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(PAIRS, fontsize=12)
    ax.set_ylabel("Cosine Similarity (stacked)",
                  fontsize=14, labelpad=8)
    ax.set_ylim(0, 0.80)
    ax.set_title("Cosine Similarity of Each Teacher's Steering Vectors with Other Teacher's",
                 fontsize=13, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # legend: one entry per A-vector (bar color) + random baseline
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = [
        Patch(facecolor=BAR_BASE_COLOR["v0"], edgecolor="white", label="Vector steering $v_0$"),
        Patch(facecolor=BAR_BASE_COLOR["v1"], edgecolor="white", label="Vector steering $v_1$"),
        Patch(facecolor=BAR_BASE_COLOR["v2"], edgecolor="white", label="Vector steering $v_2$"),
        Patch(facecolor=BAR_BASE_COLOR["v3"], edgecolor="white", label="Vector steering $v_3$"),
    ]
    # handles.append(Line2D([0], [0], color="#999", ls="--", lw=1.3, label="Random baseline"))

    ax.legend(handles=handles, bbox_to_anchor=(0.98, 1), fontsize=11,
              framealpha=0.95, edgecolor="#ddd", title_fontsize=10, ncol=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()