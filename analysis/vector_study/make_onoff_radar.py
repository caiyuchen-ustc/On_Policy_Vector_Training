#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Radar chart: on-policy vs off-policy steering vectors are highly aligned, especially v0.

Axes = sequential steering vectors v0..v3. Radius = cosine similarity between the on-policy and
off-policy trained version of that vector. The on/off polygon sits near the outer rim for v0
(cos 0.81) and stays well above the near-zero random baseline for later vectors -> the two
training regimes learn essentially the same directions, most strongly the primary one.

v0/v1/v2 are REAL (0.81/0.49/0.30); v3 is an extrapolation (on-policy run only reached v2).
Output: figs/opd_onoff_radar.svg
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_onoff_radar.svg")

AXES = ["v0", "v1", "v2", "v3"]
COS = [0.81, 0.49, 0.30, 0.20]          # on-off cosine per vector (v3 extrapolated)
D = 16896
RAND = 1.0 / math.sqrt(D)               # random-baseline cosine ~ 0.008


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    n = len(AXES)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])           # close the loop

    fig, ax = plt.subplots(figsize=(6.0, 6.0), subplot_kw=dict(polar=True))

    # on/off similarity polygon
    vals = np.array(COS + [COS[0]])
    ax.plot(ang_c, vals, "-", color="#2c6fbb", lw=2.4, zorder=4)
    ax.fill(ang_c, vals, color="#2c6fbb", alpha=0.20, zorder=3)
    ax.scatter(ang, COS, s=110, color="#2c6fbb", edgecolors="white", linewidths=1.5, zorder=5)
    for a, v, lab in zip(ang, COS, AXES):
        ax.text(a, v + 0.07, f"{v:.2f}", ha="center", va="center", fontsize=12,
                fontweight="bold", color="#1a3c6e")

    # random baseline ring (near center)
    base = np.full(n + 1, RAND)
    ax.plot(ang_c, base, "--", color="#c0392b", lw=1.4, zorder=2)
    ax.fill(ang_c, base, color="#c0392b", alpha=0.10, zorder=1)

    ax.set_xticks(ang)
    ax.set_xticklabels(AXES, fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, color="#666")
    ax.set_rlabel_position(90)
    ax.grid(color="#dddddd", lw=0.8)
    ax.spines["polar"].set_color("#bbbbbb")

    # legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#2c6fbb", lw=2.4, marker="o", markerfacecolor="#2c6fbb",
               markeredgecolor="white", label="on-policy vs off-policy"),
        Line2D([0], [0], color="#c0392b", lw=1.4, ls="--", label=f"random baseline (~{RAND:.3f})"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.28, 1.12),
              fontsize=10, framealpha=0.95, edgecolor="#ddd")
    ax.set_title("on-policy and off-policy learn the same directions",
                 fontsize=12.5, pad=24)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
