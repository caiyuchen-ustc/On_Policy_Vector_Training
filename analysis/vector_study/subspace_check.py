#!/usr/bin/env python3
"""
(1) 正交性验证：对训好的 multi basis，算 max|B B^T - I|（越小越正交）+ 有效秩。
(2) rank 扫描准备：对同层不同 N 的实验，比较其子空间；并画"最终 basis 的奇异值谱"，
    看有效维度（若 N 超过内在维度，尾部奇异值会塌）。
下游分数 vs N 的曲线需从各 N 的训练 log 提取（见 --logs），本脚本先做几何侧证明。

用法:
  python3 subspace_check.py --glob 'repre/opd/repre_qwen3-4b-opd_multi/trainable_vectors/multiv*'
"""
import os, glob, re, csv, argparse
import torch

ROOT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl"
OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def load_last(exp_dir):
    fs = sorted(glob.glob(os.path.join(exp_dir, "vectors_step_*.pt")),
                key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if not fs:
        return None, None
    d = torch.load(fs[-1], map_location="cpu")
    out = {}
    for k, v in d.items():
        t = v if torch.is_tensor(v) else v.get("vector")
        if t is None:
            continue
        out[int(k)] = t.float()   # [N,H] (multi) 或 [H] (single)
    return out, os.path.basename(fs[-1])


def analyze_basis(B):
    """B: [N,H]. 返回 正交性误差 / 各基norm / 有效秩(奇异值谱) 。"""
    N = B.shape[0]
    Bn = B / (B.norm(dim=-1, keepdim=True) + 1e-9)   # 行单位化后看方向正交性
    G = Bn @ Bn.T                                     # [N,N] gram
    off = (G - torch.eye(N)).abs()
    max_off = float(off.max())
    mean_off = float(off[~torch.eye(N, dtype=bool)].abs().mean()) if N > 1 else 0.0
    # 有效秩：对 B 做 SVD，participation ratio
    s = torch.linalg.svdvals(B)
    e = (s ** 2); e = e / e.sum()
    pr = float((e.sum() ** 2) / (e ** 2).sum())
    # 尾部能量占比（后半奇异值的能量），大N若维度过剩这里会偏小
    half = max(1, N // 2)
    tail = float(e[half:].sum()) if N > 1 else 0.0
    return dict(N=N, max_offdiag=round(max_off, 5), mean_offdiag=round(mean_off, 5),
                participation_ratio=round(pr, 3), tail_energy_frac=round(tail, 4),
                svals=[round(float(x), 3) for x in s[:8].tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="repre/opd/repre_qwen3-4b-opd_multi/trainable_vectors/multiv*")
    ap.add_argument("--tag", default="qwen3-4b")
    args = ap.parse_args()

    exps = sorted(glob.glob(os.path.join(ROOT, args.glob)))
    rows = []
    print(f"{'exp':40} {'layer':>5} {'N':>3} {'max|BBt-I|':>11} {'mean_off':>9} {'PR':>6} {'tail_E':>7}")
    for exp in exps:
        if not os.path.isdir(exp):
            continue
        vecs, last = load_last(exp)
        if not vecs:
            continue
        exp_name = os.path.basename(exp)
        for L, B in sorted(vecs.items()):
            if B.dim() == 1:   # single, 跳过
                continue
            r = analyze_basis(B)
            r["exp"] = exp_name; r["layer"] = L
            rows.append(r)
            print(f"{exp_name:40} {L:>5} {r['N']:>3} {r['max_offdiag']:>11.5f} "
                  f"{r['mean_offdiag']:>9.5f} {r['participation_ratio']:>6.2f} {r['tail_energy_frac']:>7.3f}")

    if not rows:
        print("no multi basis found for glob"); return
    csv_path = os.path.join(OUT, f"subspace_check_{args.tag}.csv")
    keys = ["exp", "layer", "N", "max_offdiag", "mean_offdiag", "participation_ratio", "tail_energy_frac", "svals"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[ok] {csv_path}")

    # 结论式打印
    all_max = max(r["max_offdiag"] for r in rows)
    print(f"\n=== 正交性结论 ===")
    print(f"所有实验所有层的 max|BB^T - I| 上界 = {all_max:.5f}")
    print(f"  ({'严格正交(fp32)' if all_max < 1e-3 else '数值正交' if all_max < 5e-2 else '偏离正交,需查'})")

    # 画: 每个 N 的 PR / tail_energy（若有多个N）
    byN = {}
    for r in rows:
        byN.setdefault(r["N"], []).append(r)
    Ns = sorted(byN)
    if len(Ns) >= 2:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pr_mean = [sum(x["participation_ratio"] for x in byN[n]) / len(byN[n]) for n in Ns]
        tail_mean = [sum(x["tail_energy_frac"] for x in byN[n]) / len(byN[n]) for n in Ns]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        ax[0].plot(Ns, pr_mean, "o-", color="purple"); ax[0].plot(Ns, Ns, "--", color="gray", label="PR=N (fully used)")
        ax[0].set_xlabel("N (subspace dim)"); ax[0].set_ylabel("participation ratio of basis")
        ax[0].set_title("effective rank vs N\n(PR<<N => extra dims underused)"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(Ns, tail_mean, "o-", color="crimson")
        ax[1].set_xlabel("N"); ax[1].set_ylabel("energy in bottom N/2 singular dirs")
        ax[1].set_title("tail energy fraction\n(low =>信号集中在少数方向)"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"figD_subspace_rank_{args.tag}.png"), dpi=130)
        print(f"[ok] figD_subspace_rank_{args.tag}.png")


if __name__ == "__main__":
    main()
