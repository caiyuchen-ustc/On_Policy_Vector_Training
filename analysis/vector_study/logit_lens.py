#!/usr/bin/env python3
"""
线 2: Logit Lens —— 把每层训练得到的 single 向量直接投影到 vocab 空间，
度量该"操控方向"在输出分布上的尖锐程度。

方法:
  向量 v 加在第 L 层输出的 residual stream 上。用 unembedding 矩阵 W_U
  (Qwen3-4B: tied embed_tokens; DeepSeek-7B: 独立 lm_head) 做 logit lens:
      logits(v) = RMSNorm(v) @ W_U^T        (近似, 忽略后续层)
  再 softmax 得分布 p, 计算:
      entropy        H(p)              高层->低(集中到少数token) 是我们要验证的
      norm_entropy   H(p)/log(V)       归一化到 [0,1], 跨模型可比
      top1_prob      max p
      top10_mass     top-10 概率之和
      part_ratio     exp(H) 有效token数
      top_tokens     top-k token 文本 (看是否是有语义的真实词)
  另做对照: 同维随机高斯向量的 norm_entropy (baseline)。

  对比两种投影:
    - raw:    直接 v @ W_U^T  (保留训练学到的尺度)
    - normed: RMSNorm(v) @ W_U^T (只看方向, 与真实前向的 RMSNorm 一致)

输出:
  out/logitlens_<model>.csv   每(exp,layer)一行
  out/logitlens_toptokens_<model>.txt  每(exp,layer)的top tokens
"""
import os, glob, re, csv, json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODELS = {
    "qwen3-4b": {
        "path": "/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B",
        "unembed_key": "model.embed_tokens.weight",   # tied
        "vec_root": "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors",
        "rms_eps": 1e-6,
    },
    "deepseek7b": {
        "path": "/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B",
        "unembed_key": "lm_head.weight",              # 独立 lm_head
        "vec_root": "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/ifevalg_opd/trainable_vectors",
        "rms_eps": 1e-6,
    },
}
OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
os.makedirs(OUT, exist_ok=True)
TOPK = 15


def load_unembed(model_cfg):
    idx = json.load(open(os.path.join(model_cfg["path"], "model.safetensors.index.json")))["weight_map"]
    key = model_cfg["unembed_key"]
    shard = idx[key]
    with safe_open(os.path.join(model_cfg["path"], shard), framework="pt", device="cpu") as f:
        W = f.get_tensor(key).float()   # [V, H]
    return W


def load_final_vecs(exp_dir):
    files = sorted(glob.glob(os.path.join(exp_dir, "vectors_step_*.pt")),
                   key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if not files:
        return {}, []
    d = torch.load(files[-1], map_location="cpu")
    out = {}
    for k, v in d.items():
        t = v if torch.is_tensor(v) else v.get("vector")
        if t is None:
            continue
        out[int(k)] = t.float().reshape(-1)
    return out, files


def rmsnorm(v, eps):
    return v / torch.sqrt((v * v).mean() + eps)


def dist_stats(logits):
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    H = float(-(p * logp).sum())
    V = logits.numel()
    top = torch.topk(p, TOPK)
    return {
        "entropy": H,
        "norm_entropy": H / torch.log(torch.tensor(float(V))).item(),
        "top1_prob": float(top.values[0]),
        "top10_mass": float(top.values[:10].sum()),
        "part_ratio": float(torch.exp(torch.tensor(H))),
        "logit_gap": float(top.values[0] / (top.values[1] + 1e-12)),  # top1/top2 prob 比
    }, top.indices.tolist(), top.values.tolist()


def main():
    for model_tag, cfg in MODELS.items():
        if not os.path.isdir(cfg["vec_root"]):
            print(f"[skip] {model_tag}: {cfg['vec_root']} missing")
            continue
        print(f"\n===== {model_tag} =====")
        W = load_unembed(cfg)   # [V,H]
        V, H = W.shape
        print(f"unembed {cfg['unembed_key']}: V={V}, H={H}")
        tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)

        # 随机向量 baseline: normed 投影的熵
        g = torch.randn(64, H)
        g = g / g.norm(dim=-1, keepdim=True)
        base_ne = []
        for i in range(64):
            s, _, _ = dist_stats(rmsnorm(g[i], cfg["rms_eps"]) @ W.T)
            base_ne.append(s["norm_entropy"])
        base_ne = sum(base_ne) / len(base_ne)
        print(f"random-vector baseline norm_entropy(normed) = {base_ne:.4f}")

        rows = []
        toptxt = [f"# {model_tag}  random_baseline_norm_entropy={base_ne:.4f}\n"]
        for exp_dir in sorted(glob.glob(os.path.join(cfg["vec_root"], "*"))):
            if not os.path.isdir(exp_dir):
                continue
            exp = os.path.basename(exp_dir)
            vecs, files = load_final_vecs(exp_dir)
            if not vecs:
                continue
            for L, v in sorted(vecs.items()):
                for mode in ("normed", "raw"):
                    vin = rmsnorm(v, cfg["rms_eps"]) if mode == "normed" else v
                    logits = vin @ W.T
                    stats, idxs, vals = dist_stats(logits)
                    rows.append({
                        "model": model_tag, "exp": exp, "layer": L, "mode": mode,
                        "vec_norm": float(v.norm()), "n_steps": len(files),
                        **{k: round(val, 5) for k, val in stats.items()},
                    })
                    if mode == "normed":
                        toks = [tok.decode([i]).replace("\n", "\\n") for i in idxs]
                        toptxt.append(f"[{exp} | L{L} | normed | H_norm={stats['norm_entropy']:.3f} "
                                      f"top1={stats['top1_prob']:.3f}] " + " | ".join(toks[:TOPK]))
        # 写出
        cp = os.path.join(OUT, f"logitlens_{model_tag}.csv")
        with open(cp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(OUT, f"logitlens_toptokens_{model_tag}.txt"), "w") as fh:
            fh.write("\n".join(toptxt))
        print(f"[ok] {cp}  ({len(rows)} rows)")

        # 打印 normed 模式下按层的 norm_entropy 趋势(取每层跨exp均值)
        from collections import defaultdict
        agg = defaultdict(list)
        for r in rows:
            if r["mode"] == "normed":
                agg[r["layer"]].append(r["norm_entropy"])
        print(f"  {'layer':>5} {'mean_norm_entropy':>18} {'n_exp':>6}  (baseline={base_ne:.3f})")
        for L in sorted(agg):
            xs = agg[L]
            print(f"  {L:>5} {sum(xs)/len(xs):>18.4f} {len(xs):>6}")


if __name__ == "__main__":
    main()
