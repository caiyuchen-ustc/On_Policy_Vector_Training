#!/usr/bin/env python3
"""汇总出图: 验证 高层向量->输出token操纵, 中低层->内部模式 的假设。"""
import os, csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
os.makedirs(FIG, exist_ok=True)


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def layer_mean(rows, key, mode=None, model=None):
    agg = defaultdict(list)
    for r in rows:
        if mode and r.get("mode") != mode:
            continue
        if model and r.get("model") != model:
            continue
        try:
            agg[int(r["layer"])].append(float(r[key]))
        except (ValueError, KeyError):
            continue
    xs = sorted(agg)
    return xs, [sum(agg[L]) / len(agg[L]) for L in xs]


# ---- 图1: logit-lens norm_entropy vs layer (两模型) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (model, csvf, base, nlayer) in zip(axes, [
    ("qwen3-4b", "logitlens_qwen3-4b.csv", 0.9498, 36),
    ("deepseek7b", "logitlens_deepseek7b.csv", 0.9594, 28),
]):
    rows = read_csv(os.path.join(OUT, csvf))
    xs, ys = layer_mean(rows, "norm_entropy", mode="normed")
    ax.plot(xs, ys, "o-", color="crimson", label="trained vector (normed)")
    ax.axhline(base, ls="--", color="gray", label=f"random-vector baseline={base:.3f}")
    ax.set_xlabel("layer index"); ax.set_ylabel("logit-lens norm. entropy")
    ax.set_title(f"{model}: vector→vocab sharpness vs depth")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # 标注高层坍缩
    lo = [ (L,y) for L,y in zip(xs,ys) if y < base-0.03 ]
    for L,y in lo:
        ax.annotate("", xy=(L,y), xytext=(L,base),
                    arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.4))
fig.suptitle("Logit-Lens: low/mid-layer vectors are vocab-agnostic (=模式), high-layer vectors collapse onto few tokens (=token操纵)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_logitlens_entropy_vs_depth.png"), dpi=130)
print("saved fig1")

# ---- 图2: top1_prob vs layer (越高=越被单token主导) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for model, csvf, c in [("qwen3-4b","logitlens_qwen3-4b.csv","crimson"),
                       ("deepseek7b","logitlens_deepseek7b.csv","navy")]:
    rows = read_csv(os.path.join(OUT, csvf))
    xs, ys = layer_mean(rows, "top1_prob", mode="normed")
    ax.plot(xs, ys, "o-", color=c, label=model)
ax.set_xlabel("layer index"); ax.set_ylabel("logit-lens top-1 prob")
ax.set_title("High-layer vectors concentrate on a single output token")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_top1prob_vs_depth.png"), dpi=130)
print("saved fig2")

# ---- 图3: 轨迹指标 vs layer (straightness & final_norm) 用受控实验 24-35 & 10-25 ----
traj = read_csv(os.path.join(OUT, "trajectory_summary.csv"))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
# 只用跨越大层范围的单实验, 控制lr一致
for exp, c, lab in [("singlev_layers24-35_lr1e-3","crimson","L24-35 (high)"),
                    ("singlev_layers10-25_lr5e-5","navy","L10-25 (mid)"),
                    ("singlev_layers0-11_lr1e-3","green","L0-11 (low)"),
                    ("singlev_layers12-23_lr1e-3","orange","L12-23 (mid)")]:
    sub = [r for r in traj if r["exp"] == exp and r["model"]=="qwen3-4b-opd"]
    if not sub: continue
    sub.sort(key=lambda r:int(r["layer"]))
    L=[int(r["layer"]) for r in sub]
    axes[0].plot(L,[float(r["straightness"]) for r in sub],"o-",color=c,label=lab)
    axes[1].plot(L,[float(r["final_norm"]) for r in sub],"o-",color=c,label=lab)
axes[0].set_title("qwen3-4b: trajectory straightness vs layer\n(net displacement / path length; 低=绕路)")
axes[0].set_xlabel("layer"); axes[0].set_ylabel("straightness"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].set_title("qwen3-4b: final vector norm vs layer")
axes[1].set_xlabel("layer"); axes[1].set_ylabel("final ‖v‖"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_trajectory_vs_depth.png"), dpi=130)
print("saved fig3")
print("figs in", FIG)
