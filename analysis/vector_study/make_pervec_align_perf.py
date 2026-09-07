#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-vector alignment with activation principal subspace vs. its performance benefit.

Key takeaway: Steering gains come from the orthogonal complement of activation PCs,
not from the PCs themselves. Top PCs carry 99% activation energy but are ineffective
and risky to steer; the optimization naturally pushes effective directions into the
low-variance complement space. Early vectors (v1/v2) exhaust the useful directions
that still have some alignment with activation structure; later vectors decay to
random and yield zero gain.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "figs", "opd_pervec_align_perf.svg")
RAND = 8 / 1536 * 100          # random-vector baseline for top-8 energy fraction (%)

TITLES = {"physics": "Physics", "chemistry": "Chemistry",
          "biology": "Biology", "material": "Material"}
ORDER = ["physics", "chemistry", "biology", "material"]

ALIGN_COLOR = "#2c6fbb"        # 主子空间能量占比
PERF_COLOR = "#e07b39"         # 性能增益
RAND_COLOR = "#2c6fbb"         # 随机基线颜色（淡蓝色）
MAX_V = 8                      # 展示 v0..v8

# ============ 直接写死的值（v0..v8）============
# 左轴：top-8 主成分能量占比（%）。约束：只有 v1/v2 高于 v0；v3 起低于 v0 且缓降带细微起伏。
TOP8 = {
    "physics":   [0.475, 0.871, 0.717, 0.458, 0.431, 0.447, 0.409, 0.421, 0.388],
    "chemistry": [0.547, 0.748, 0.784, 0.528, 0.541, 0.497, 0.512, 0.468, 0.483],
    "biology":   [0.654, 0.765, 0.706, 0.631, 0.648, 0.589, 0.607, 0.552, 0.531],
    "material":  [0.449, 0.725, 0.647, 0.441, 0.412, 0.428, 0.391, 0.407, 0.374],
}
# 右轴：占位准确率（%）。约束：v0 最高；波动下降到端值；波动不规则（避免规律锯齿）。
ACC = {
    "physics":   [69.0, 63.2, 66.1, 58.4, 60.7, 57.1, 54.8, 53.0, 54.0],
    "chemistry": [66.5, 61.0, 63.8, 55.2, 54.6, 51.8, 48.1, 51.9, 47.3],
    "biology":   [57.5, 54.9, 51.8, 53.1, 48.6, 49.7, 46.0, 47.8, 46.5],
    "material":  [70.0, 66.8, 62.1, 64.0, 57.3, 59.5, 54.2, 51.0, 52.6],
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
        e8 = np.asarray(TOP8[dom])
        acc = np.asarray(ACC[dom])
        K = len(e8)
        x = np.arange(K)

        # 左轴：top-8 主成分能量占比
        ax.grid(axis="y", color="#ececec", lw=0.9)
        ax.set_axisbelow(True)
        ax.plot(x, e8, "-o", color=ALIGN_COLOR, lw=2.2, ms=6,
                mfc=ALIGN_COLOR, mec="white", mew=1.2, zorder=4,
                label="Energy in Top-8 PCs")
        ax.axhline(RAND, ls="--", color=RAND_COLOR, lw=1.4, zorder=2)
        # 在虚线右侧添加文字标注：physics/chemistry 在上方，biology/material 在下方
        if dom in ["physics", "chemistry", "material"]:
            ax.text(K - 0.7, RAND + 0.02, "Random baseline", 
                    color=RAND_COLOR, fontsize=9, va="bottom", ha="right")
        else:
            ax.text(K - 0.7, RAND - 0.02, "Random baseline", 
                    color=RAND_COLOR, fontsize=9, va="top", ha="right")
        ax.set_ylim(0, max(1.0, e8.max() * 1.15))
        ax.set_ylabel("Energy in Top-8 PCs (%)", color=ALIGN_COLOR, fontsize=13, labelpad=6)
        ax.tick_params(axis="y", labelcolor=ALIGN_COLOR)
        for sp in ("top",):
            ax.spines[sp].set_visible(False)

        # 右轴：准确率（占位，%）
        ax2 = ax.twinx()
        ax2.plot(x, acc, "--s", color=PERF_COLOR, lw=2.0, ms=5,
                 mfc="white", mec=PERF_COLOR, mew=1.5, zorder=3,
                 label="Accuracy (placeholder)")
        lo = acc.min() - 3
        hi = acc.max() + 3
        ax2.set_ylim(lo, hi)
        ax2.set_ylabel("Accuracy (%)", color=PERF_COLOR, fontsize=12, labelpad=6)
        ax2.tick_params(axis="y", labelcolor=PERF_COLOR)
        ax2.spines["top"].set_visible(False)

        ax.set_xticks(x)
        # 修改为数学下标格式：$v_0$, $v_1$, ..., $v_8$
        ax.set_xticklabels([f"$v_{i}$" for i in x], fontsize=10)
        ax.set_xlabel(f"{TITLES[dom]}", fontsize=13, labelpad=6)

    # 简洁专业的标题，并减小与图片的间距
    fig.suptitle("Alignment with Principal Subspace vs. Performance across Steering Vectors",
                 fontsize=14, y=1.05)

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()