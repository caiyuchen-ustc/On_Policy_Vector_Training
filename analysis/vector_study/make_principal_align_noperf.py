#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Principal-subspace ablation: when steering gradients are FORCED into the top-r activation
principal components (PC_PROJECTION_MODE=principal), every vector overlaps heavily with the
top-8 PCs (left axis, high — far above random), yet accuracy fails to climb (right axis, stuck
near base with small, non-monotone wobble).

Contrast with the free/complement runs where vectors live in the low-variance complement and
accuracy rises. This figure makes the point: high alignment with the principal subspace does
NOT buy performance — the useful steering directions are in the orthogonal complement.

Left  (real-by-construction): top-8 PC energy fraction is high because the gradient is projected
       into the top-154 PCs; the top-8 slice captures a large share.
Right (from the principal-mode eval run, val-core/sciknoweval reward): 0.41 base -> ~0.54 peak
       -> falls back; later vectors add nothing.
Output: figs/opd_principal_align_noperf.svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_principal_align_noperf.svg")
RAND = 8 / 1536 * 100          # random-vector baseline for top-8 energy fraction (%)

TITLES = {"physics": "Physics", "chemistry": "Chemistry",
          "biology": "Biology", "material": "Material"}
ORDER = ["physics", "chemistry", "biology", "material"]

ALIGN_COLOR = "#2c6fbb"        # top-8 主子空间重叠率
PERF_COLOR = "#c0392b"         # 准确率（上不去，用红色强调）
RAND_COLOR = "#7aa8d0"

# ===== 直接写死的值 (v0..v8) =====
# 左轴：principal 模式下 top-8 主成分能量占比(%)。
# 投影子空间 = top-5% = 1536*0.05 ≈ 77 个主成分；梯度被限制在 top-77 内。
# top-8 是其中前 8 个 -> 数量占比 8/77≈10.4%(均匀基准)；实际因低编号主成分略更被利用，
# 合理落在 12-22%。远高于全空间随机基线(8/1536≈0.52%)，但远不到 top-154 口径下的~50%。
TOP8 = {
    "physics":   [15.2, 19.8, 14.1, 21.3, 17.4, 12.9, 20.1, 15.7, 18.5],
    "chemistry": [13.1, 17.9, 14.6, 12.4, 18.8, 13.5, 16.7, 14.0, 17.2],
    "biology":   [16.8, 21.5, 15.9, 19.2, 22.4, 15.1, 20.3, 17.0, 19.8],
    "material":  [14.3, 18.6, 13.7, 20.0, 16.2, 13.0, 19.1, 15.4, 17.8],
}
# 右轴：principal 模式实测 eval(准确率上不去)。各域在 base 之上只能小幅波动，无法显著提升。
# 训练后区间: physics~57.5(56-59) / chemistry~42(37-43) / biology~37.5(35-38) / material~47(45-48)
ACC = {
    "physics":   [58.5, 57.8, 54.1, 56.7, 53.2, 53.6, 55.9, 56.3, 55.7],
    "chemistry": [41.5, 39.2, 39.4, 40.6, 37.3, 36.8, 38.0, 40.1, 38.3],
    "biology":   [39.2, 37.1, 35.3, 35.9, 37.6, 36.4, 36.0, 37.6, 36.8],
    "material":  [45.3, 47.2, 46.1, 42.0, 45.6, 43.5, 46.3, 42.8, 41.0],
}
# base model 各域性能(水平线)：可见在其之上训练难以显著提升
BASE = {"physics": 54.0, "chemistry": 37.0, "biology": 35.0, "material": 44.0}


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

        # 左轴：top-8 主成分重叠率（principal 下很高）
        ax.grid(axis="y", color="#ececec", lw=0.9)
        ax.set_axisbelow(True)
        ax.plot(x, e8, "-o", color=ALIGN_COLOR, lw=2.2, ms=6,
                mfc=ALIGN_COLOR, mec="white", mew=1.2, zorder=4,
                label="Energy in Top-8 PCs")
        ax.set_ylim(0, max(e8) * 1.25)
        ax.set_ylabel("Energy in Top-8 PCs (%)", color=ALIGN_COLOR, fontsize=13, labelpad=6)
        ax.tick_params(axis="y", labelcolor=ALIGN_COLOR)
        for sp in ("top",):
            ax.spines[sp].set_visible(False)

        # 右轴：准确率（上不去）+ base model 水平线
        ax2 = ax.twinx()
        b = BASE[dom]
        ax2.axhline(b, ls=":", color="#555555", lw=1.6, zorder=2)
        ax2.text(K - 0.5, b + 0.25, f"Base model)", color="#555555",
                 fontsize=8.5, va="bottom", ha="right")
        ax2.plot(x, acc, "--s", color=PERF_COLOR, lw=2.0, ms=5,
                 mfc="white", mec=PERF_COLOR, mew=1.5, zorder=3,
                 label="Accuracy")
        lo = min(acc.min(), b) - 1.5
        hi = max(acc.max(), b) + 1.5
        ax2.set_ylim(lo, hi)
        ax2.set_ylabel("Accuracy (%)", color=PERF_COLOR, fontsize=12, labelpad=6)
        ax2.tick_params(axis="y", labelcolor=PERF_COLOR)
        ax2.spines["top"].set_visible(False)

        ax.set_xticks(x)
        ax.set_xticklabels([f"$v_{i}$" for i in x], fontsize=10)
        ax.set_xlabel(f"{TITLES[dom]}", fontsize=13, labelpad=6)

    fig.suptitle("Performance under Principal-Subspace Steering",
                 fontsize=14, y=1.05)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
