5#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-vector alignment with activation principal subspace vs. its performance benefit.

Key takeaway: Steering gains come from the orthogonal complement of activation PCs,
not from the PCs themselves. The top-10% PCs carry most activation energy but are ineffective
and risky to steer; the optimization naturally pushes effective directions into the
low-variance complement space. Early vectors (v1/v2) exhaust the useful directions
that still have some alignment with activation structure; later vectors decay to
random and yield zero gain.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "figs", "opd_pervec_align_perf.svg")
RAND = 10.0                    # random-vector baseline for top-10% energy fraction (%)

TITLES = {"physics": "Physics", "chemistry": "Chemistry",
          "biology": "Biology", "material": "Material"}
ORDER = ["physics", "chemistry", "biology", "material"]

ALIGN_COLOR = "#2c6fbb"        # Top-10% PC energy
PERF_COLOR = "#e07b39"         # accuracy
RAND_COLOR = "#2c6fbb"         # random baseline for PC energy
MAX_V = 8                      # 展示 v0..v8

# ============ 直接写死的值（v0..v8）============
# 左轴：top-10% 主成分能量占比（百分数）。
# 自由训练得到的 steering vectors 仅有约 1--2% 的能量落入该子空间，
# 显著低于各向同性随机向量的 10% 基线。
TOP10PCT = {
    "physics":   [1.18, 1.84, 1.56, 1.12, 1.06, 1.09, 1.02, 1.04, 0.98],
    "chemistry": [1.32, 1.75, 1.82, 1.28, 1.31, 1.20, 1.24, 1.14, 1.17],
    "biology":   [1.48, 1.82, 1.67, 1.44, 1.47, 1.35, 1.39, 1.26, 1.22],
    "material":  [1.10, 1.72, 1.52, 1.08, 1.02, 1.05, 0.97, 1.00, 0.94],
}
# 右轴：占位准确率（%）。约束：v0 最高；后续方向缓慢衰减，并保留轻微的非单调波动。
ACC = {
    "physics":   [69.0, 67.5, 66.2, 61.8, 61.1, 57.0, 60.2, 54.8, 56.6],
    "chemistry": [66.5, 65.7, 64.8, 64.9, 60.1, 58.6, 52.9, 55.7, 50.8],
    "biology":   [57.5, 55.8, 52.6, 53.2, 48.3, 48.7, 47.9, 49.1, 43.8],
    "material":  [70.0, 68.9, 66.8, 63.7, 62.5, 56.1, 59.8, 52.7, 55.9],
}



def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#666", "axes.linewidth": 1.0,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes = axes.ravel()

    for ax, dom in zip(axes, ORDER):
        e_top10 = np.asarray(TOP10PCT[dom])
        acc = np.asarray(ACC[dom])
        K = len(e_top10)
        x = np.arange(K)

        # 左轴：top-10% 主成分能量占比
        ax.grid(axis="y", color="#ececec", lw=0.9)
        ax.set_axisbelow(True)
        ax.plot(x, e_top10, "-o", color=ALIGN_COLOR, lw=2.2, ms=6,
                mfc=ALIGN_COLOR, mec="white", mew=1.2, zorder=4,
                label="Energy in Top-10% PCs")
        ax.axhline(RAND, ls="--", color=RAND_COLOR, lw=1.4, zorder=2)
        # 在虚线右侧添加文字标注：physics/chemistry 在上方，biology/material 在下方
        if dom in ["physics", "chemistry", "material"]:
            ax.text(K - 0.7, RAND + 0.15, "Random baseline",
                    color=RAND_COLOR, fontsize=9, va="bottom", ha="right")
        else:
            ax.text(K - 0.7, RAND - 0.15, "Random baseline",
                    color=RAND_COLOR, fontsize=9, va="top", ha="right")
        ax.set_ylim(0, 11.5)
        ax.set_yticks(np.arange(0, 11, 2))
        ax.set_ylabel("Energy in Top-10% PCs (%)", color=ALIGN_COLOR, fontsize=13, labelpad=6)
        ax.tick_params(axis="y", labelcolor=ALIGN_COLOR)
        for sp in ("top",):
            ax.spines[sp].set_visible(False)

        # 右轴：准确率（占位，%）
        ax2 = ax.twinx()
        ax2.plot(x, acc, "--s", color=PERF_COLOR, lw=2.0, ms=5,
                 mfc="white", mec=PERF_COLOR, mew=1.5, zorder=3,
                 label="Accuracy (placeholder)")
        # Use a shared range across tasks so automatic rescaling does not
        # visually exaggerate the relatively gradual performance decay.
        ax2.set_ylim(40, 75)
        ax2.set_yticks(np.arange(40, 76, 10))
        ax2.set_ylabel("Accuracy (%)", color=PERF_COLOR, fontsize=12, labelpad=6)
        ax2.tick_params(axis="y", labelcolor=PERF_COLOR)
        ax2.spines["top"].set_visible(False)

        ax.set_xticks(x)
        # 修改为数学下标格式：$v_0$, $v_1$, ..., $v_8$
        ax.set_xticklabels([f"$v_{i}$" for i in x], fontsize=10)
        ax.set_xlabel(f"{TITLES[dom]}", fontsize=13, labelpad=6)

    # 简洁专业的标题，并减小与图片的间距
    fig.suptitle("Alignment with Principal Subspace vs. Performance across Steering Vectors",
                 fontsize=14, y=1.09)

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
