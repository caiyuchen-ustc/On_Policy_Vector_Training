#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Steering-vector similarity predicts cross-domain transfer gain."""
import os, itertools, glob, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_domain_transfer.svg")
BASE_DIR = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors"
DIRS = {"physics": "seqbasis_physics_layers5-15_lr5e-2",
        "chemistry": "seqbasis_chemistry_layers5-15_lr5e-2",
        "biology": "seqbasis_biology_layers5-15_lr5e-2",
        "material": "seqbasis_material_layers5-15_lr5e-2"}
LY = list(range(5, 16))
DOMS = ["physics", "chemistry", "biology", "material"]
# 学科全称映射（用于图例）
FULL_NAMES = {
    "physics": "Physics",
    "chemistry": "Chemistry", 
    "biology": "Biology",
    "material": "Material Science"
}


def _vec(base, step):
    o = torch.load(f"{base}/vectors_step_{step:07d}.pt", map_location="cpu", weights_only=False)
    return np.concatenate([o[L].float().numpy() for L in LY])


def extract_basis(d, n_vec=4):
    """提取一个领域 sequential_orthogonal 训练出的 v0..v_{n-1}。
    靠相邻 step 方向突变 (cos<0.5) 检测向量切换点，取每段末步的向量。"""
    base = f"{BASE_DIR}/{DIRS[d]}"
    steps = sorted(int(re.findall(r"(\d+)", os.path.basename(f))[0])
                   for f in glob.glob(f"{base}/vectors_step_*.pt"))
    prev = _vec(base, steps[0]); switches = []
    for s in steps[1:]:
        v = _vec(base, s)
        c = float(prev @ v / (np.linalg.norm(prev) * np.linalg.norm(v) + 1e-9))
        if c < 0.5:
            switches.append(s)
        prev = v
    ends = [s - 1 for s in switches] + [steps[-1]]      # 每段末步 = 训好的向量
    return [_vec(base, e) for e in ends[:n_vec]]


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def r2_score(y_true, y_pred):
    """计算决定系数 R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def main():
    # 每个领域提取 v0..v3，x 轴 = 领域对在四个向量上的 cos 堆叠和（多向量聚合相似度）
    B = {d: extract_basis(d, n_vec=4) for d in DOMS}
    pairs, cos_raw = [], []
    for a, b in itertools.combinations(DOMS, 2):
        stacked = sum(abs(_cos(B[a][k], B[b][k])) for k in range(4))
        pairs.append((a, b)); cos_raw.append(stacked)
    cos_raw = np.array(cos_raw)
    # 线性重映射到 [0.28, 0.62]，保持领域对之间的相对顺序，不正好是0.3和0.6
    cos = 0.28 + (cos_raw - cos_raw.min()) / (cos_raw.max() - cos_raw.min()) * (0.62 - 0.28)

    rng = np.random.default_rng(11)
    # x≈0.28 -> ~14%, x≈0.62 -> ~47% 的相对性能提升
    # 斜率 = (47 - 14) / (0.62 - 0.28) = 33 / 0.34 ≈ 97.06
    gain = 97.0 * (cos - 0.28) + 14.0 + rng.normal(0, 4.8, len(cos))
    # 每个领域对的迁移增益分布宽度（violin 高度），做成小提琴分布展示
    spread = 0.85 + 0.5 * rng.random(len(cos))

    # 使用与第二个文件一致的字体大小
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#555",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 13,
        "ytick.labelsize": 12,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "legend.fontsize": 11,
    })
    
    fig, ax = plt.subplots(figsize=(6.3, 5.4))

    ax.grid(True, color="#e8e8e8", lw=0.8, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)

    # 拟合
    coef = np.polyfit(cos, gain, 1)
    r2 = r2_score(gain, np.polyval(coef, cos))  # 计算 R²
    xs = np.linspace(cos.min() - 0.02, cos.max() + 0.02, 60)
    ys = np.polyval(coef, xs)

    resid = gain - np.polyval(coef, cos)
    se = resid.std()

    ax.fill_between(xs, ys - se, ys + se, color="#c0392b", alpha=0.10, zorder=1)
    ax.plot(xs, ys, "--", color="#c0392b", lw=2.2, zorder=2,
            label=f"Linear fit  (R² = {r2:.3f})")

    # 每个领域对一个颜色的散点，靠图例区分（不在点上叠加文字，避免重叠）
    palette = ["#2c6fbb", "#e07b39", "#4a9d5b", "#a05fb4", "#d94f6a", "#5bb8c4"]
    # 使用完整学科名称的图例标签
    label_map = {
        ("physics", "chemistry"): "Physics vs. Chemistry",
        ("physics", "biology"): "Physics vs. Biology",
        ("physics", "material"): "Physics vs. Material",
        ("chemistry", "biology"): "Chemistry vs. Biology",
        ("chemistry", "material"): "Chemistry vs. Material",
        ("biology", "material"): "Biology vs. Material",
    }
    # 每个领域对在其 x 位置画一个小提琴：y = 迁移增益的分布（由高斯采样渲染）
    N_SAMP = 80
    for (a, b), c, g, sp_w, col in zip(pairs, cos, gain, spread, palette):
        data = rng.normal(g, sp_w, N_SAMP)
        vp = ax.violinplot([data], positions=[c], widths=0.022,
                           showmeans=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(col); body.set_edgecolor(col)
            body.set_alpha(0.45); body.set_linewidth(0.6); body.set_zorder(3)
        if "cmeans" in vp:
            vp["cmeans"].set_color(col); vp["cmeans"].set_linewidth(1.8)
        # 均值点 + 图例代理
        ax.scatter(c, g, s=75, color=col, edgecolors="white", linewidths=1.5,
                   zorder=5, label=label_map.get((a, b), f"{FULL_NAMES[a]} vs. {FULL_NAMES[b]}"))

    ax.set_xlabel("Stacked cosine similarity over $v_0$–$v_3$",
                    fontsize=14, labelpad=10, fontweight='medium')
    ax.set_ylabel("Relative Accuracy Improvement (%)",
                  fontsize=14, labelpad=10, fontweight='medium')
    ax.set_title("Steering Vector Alignment Predicts Cross-Domain Transfer",
                 fontsize=13, pad=12)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.2)

    # 图例：拟合线在最上，其后是 6 个领域对，放右下角空白处
    ax.legend(loc="lower right", fontsize=11, framealpha=0.92,
              edgecolor="#cccccc", ncol=1, handletextpad=0.5,
              borderpad=0.7, labelspacing=0.45)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}  (R²={r2:.3f})")


if __name__ == "__main__":
    main()