#!/usr/bin/env python3
"""
线 1: 单向量训练轨迹演化分析。

对每个实验(层范围+lr)、每一层，逐 step 加载保存的向量，计算:
  - norm            ‖v_t‖
  - cos_final       cos(v_t, v_final)         方向是否收敛到最终方向
  - cos_prev        cos(v_t, v_{t-1})         相邻 step 方向一致性(1=平滑)
  - step_size       ‖v_t - v_{t-1}‖           绝对更新幅度
  - rel_step        ‖v_t - v_{t-1}‖ / ‖v_t‖   相对更新幅度
  - path_len        累积 Σ‖Δ‖ (到当前 step)
  - straightness    ‖v_t - v_0‖ / path_len    净位移/路程, 越小越绕路

输出:
  out/trajectory_<tag>.csv   每个实验一份逐step逐layer的明细
  out/trajectory_summary.csv 每个(实验,层)的汇总(最终norm、最终cos_prev、总straightness等)
"""
import os, glob, re, csv
import torch

VEC_ROOTS = {
    "qwen3-4b-opd": "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors",
    "ifevalg-opd":  "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/ifevalg_opd/trainable_vectors",
    "ifevalg-rl":   "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/ifevalg_rl/trainable_vectors",
}
OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
os.makedirs(OUT, exist_ok=True)


def load_vec_dict(path):
    """返回 {layer_idx: 1D float tensor}. 兼容 raw tensor 和 {'vector':..,'alpha':..} 两种格式。"""
    d = torch.load(path, map_location="cpu")
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            vec = v.float().reshape(-1)
        elif isinstance(v, dict):
            t = v.get("vector")
            if t is None:
                continue
            t = t.float().reshape(-1)
            a = v.get("alpha")
            if a is not None:
                # 有 alpha 时, 实际注入向量 = normalize(v)*alpha; 但我们研究 raw 方向+尺度,
                # 这里保留 raw vector, alpha 单独不合并(single raw 模式本就无 alpha)。
                pass
            vec = t
        else:
            continue
        out[int(k)] = vec
    return out


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def analyze_experiment(model_tag, exp_name, exp_dir):
    files = sorted(glob.glob(os.path.join(exp_dir, "vectors_step_*.pt")),
                   key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if len(files) < 3:
        return [], []
    steps = [int(re.search(r"step_(\d+)", p).group(1)) for p in files]

    # 逐 step 加载, 按层组织成序列
    seq = {}   # layer -> list[tensor] over steps
    for f in files:
        vd = load_vec_dict(f)
        for L, vec in vd.items():
            seq.setdefault(L, []).append(vec)

    rows = []       # 明细
    summary = []    # 每层一行
    for L, vecs in sorted(seq.items()):
        if len(vecs) != len(files):
            # 某些step缺该层, 跳过对齐问题
            n = min(len(vecs), len(files))
            vecs = vecs[:n]
            local_steps = steps[:n]
        else:
            local_steps = steps
        v_final = vecs[-1]
        v0 = vecs[0]
        path_len = 0.0
        for i, (st, v) in enumerate(zip(local_steps, vecs)):
            if i == 0:
                d_prev = 0.0
                cos_prev = 1.0
            else:
                delta = v - vecs[i - 1]
                d_prev = float(delta.norm())
                cos_prev = cos(v, vecs[i - 1])
            path_len += d_prev
            nrm = float(v.norm())
            rows.append({
                "model": model_tag, "exp": exp_name, "layer": L, "step": st,
                "norm": nrm,
                "cos_final": cos(v, v_final),
                "cos_prev": cos_prev,
                "step_size": d_prev,
                "rel_step": d_prev / nrm if nrm > 1e-9 else 0.0,
                "path_len": path_len,
            })
        net_disp = float((v_final - v0).norm())
        summary.append({
            "model": model_tag, "exp": exp_name, "layer": L,
            "n_steps": len(vecs),
            "final_norm": float(v_final.norm()),
            "final_cos_prev": rows[-1]["cos_prev"],
            "mean_cos_prev": sum(r["cos_prev"] for r in rows if r["layer"] == L and r["step"] > local_steps[0]) / max(1, len(vecs) - 1),
            "net_disp": net_disp,
            "path_len": path_len,
            "straightness": net_disp / path_len if path_len > 1e-9 else 0.0,
            # step 到达 cos_final>0.95 (方向基本定型) 的步数, 归一化到 [0,1]
            "converge_frac": next((i / len(vecs) for i, r in
                                   enumerate(x for x in rows if x["layer"] == L)
                                   if r["cos_final"] > 0.95), 1.0),
        })
    return rows, summary


def main():
    all_summary = []
    for model_tag, root in VEC_ROOTS.items():
        if not os.path.isdir(root):
            print(f"[skip] {root} not found")
            continue
        for exp_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(exp_dir):
                continue
            exp = os.path.basename(exp_dir)
            rows, summary = analyze_experiment(model_tag, exp, exp_dir)
            if not rows:
                continue
            tag = f"{model_tag}__{exp}"
            csv_path = os.path.join(OUT, f"trajectory_{tag}.csv")
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            all_summary.extend(summary)
            print(f"[ok] {tag}: {len(rows)} rows, layers={sorted(set(r['layer'] for r in rows))}")

    if all_summary:
        sp = os.path.join(OUT, "trajectory_summary.csv")
        with open(sp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_summary[0].keys()))
            w.writeheader(); w.writerows(all_summary)
        print(f"\n[summary] {sp}  ({len(all_summary)} (exp,layer) rows)")
        # 打印按层排序的关键对比
        print("\n=== 关键汇总 (按 model, layer) ===")
        print(f"{'model':16} {'exp':30} {'layer':>5} {'final_norm':>10} {'straight':>8} {'mean_cosprev':>12}")
        for s in sorted(all_summary, key=lambda x: (x["model"], x["layer"])):
            print(f"{s['model']:16} {s['exp']:30} {s['layer']:>5} {s['final_norm']:>10.3f} "
                  f"{s['straightness']:>8.3f} {s['mean_cos_prev']:>12.4f}")


if __name__ == "__main__":
    main()
