#!/usr/bin/env python3
"""
任务3: 沿 step 的 logit-lens 熵演化 (模式->token 退化过程)
任务4: W_U 行空间能量分解 (向量有多少能量落在"输出token空间")

任务4 方法:
  W_U [V,H] 是 unembedding。它的行张成一个 <=H 维子空间, 但 V>>H 时行空间几乎充满 R^H,
  直接用秩不行。改用"主 logit 方向": 对 W_U 做 SVD, 取右奇异向量 V_r [H,r] (top-r 主成分,
  即 unembedding 能量最集中的 r 维方向 = 真正区分 token 的主方向)。
  向量 v 落在该子空间的能量占比:
      proj_energy(r) = ‖V_r V_r^T v‖^2 / ‖v‖^2
  低层若 proj_energy 低 -> 向量在"非token主方向"里; 高层若高 -> 落在token判别主方向。
  对照: 随机向量的 proj_energy(r) ≈ r/H。
"""
import os, glob, re, csv, json
from collections import defaultdict
import torch
from safetensors import safe_open

OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

MODELS = {
    "qwen3-4b": dict(
        path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B",
        unembed_key="model.embed_tokens.weight",
        vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors",
        # 选一个跨层最广的受控实验做 step 演化
        step_exp="singlev_layers24-35_lr1e-3",
        step_exp_mid="singlev_layers10-25_lr5e-5",
        rms_eps=1e-6,
    ),
}
RANKS = [8, 32, 128, 512]  # top-r 主成分维度


def load_unembed(cfg):
    idx = json.load(open(os.path.join(cfg["path"], "model.safetensors.index.json")))["weight_map"]
    k = cfg["unembed_key"]; shard = idx[k]
    with safe_open(os.path.join(cfg["path"], shard), framework="pt", device="cpu") as f:
        return f.get_tensor(k).float()


def load_vecs(pt):
    d = torch.load(pt, map_location="cpu")
    out = {}
    for k, v in d.items():
        t = v if torch.is_tensor(v) else v.get("vector")
        if t is not None:
            out[int(k)] = t.float().reshape(-1)
    return out


def rmsnorm(v, eps): return v / torch.sqrt((v*v).mean()+eps)


def norm_entropy(logits):
    p = torch.softmax(logits, -1); lp = torch.log_softmax(logits, -1)
    H = float(-(p*lp).sum())
    return H / torch.log(torch.tensor(float(logits.numel()))).item()


def main():
    cfg = MODELS["qwen3-4b"]
    W = load_unembed(cfg); V, H = W.shape
    print(f"W_U: V={V} H={H}")

    # ---- SVD 取 top-r 主方向 (右奇异向量) ----
    # W = U S Vh ; Vh[:r] 的行是 R^H 中 unembedding 能量最大的 r 个方向
    print("computing SVD of W_U (this takes a moment)...")
    # 用 float32, V太大用 lowrank 近似取最大 rank
    maxr = max(RANKS)
    U, S, Vh = torch.svd_lowrank(W, q=maxr+16, niter=4)   # W ≈ U diag(S) Vh^T, Vh:[H,q]
    Vh_main = Vh[:, :maxr]   # [H, maxr] 列是主方向
    print("singular value decay (top): ", [round(float(x),1) for x in S[:8]], "...", [round(float(x),1) for x in S[maxr-3:maxr]])

    def proj_energy(v, r):
        Vr = Vh_main[:, :r]                 # [H,r]
        c = Vr.T @ v                        # [r]
        return float((c@c) / (v@v + 1e-12))

    # ============ 任务4: 每层最终向量的能量分解 ============
    rows4 = []
    for exp_dir in sorted(glob.glob(os.path.join(cfg["vec_root"], "*"))):
        if not os.path.isdir(exp_dir): continue
        exp = os.path.basename(exp_dir)
        fs = sorted(glob.glob(exp_dir+"/vectors_step_*.pt"),
                    key=lambda p:int(re.search(r"step_(\d+)",p).group(1)))
        if not fs: continue
        vecs = load_vecs(fs[-1])
        for L, v in sorted(vecs.items()):
            row = {"model":"qwen3-4b","exp":exp,"layer":L,"vec_norm":round(float(v.norm()),4)}
            for r in RANKS:
                row[f"projE_r{r}"] = round(proj_energy(v, r), 5)
                row[f"projE_r{r}_over_random"] = round(proj_energy(v, r) / (r/H), 3)
            rows4.append(row)
    with open(os.path.join(OUT,"wU_energy.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows4[0].keys())); w.writeheader(); w.writerows(rows4)
    print(f"[task4] wU_energy.csv  {len(rows4)} rows")
    # 按层聚合打印 (r=128)
    agg=defaultdict(list)
    for r in rows4: agg[r["layer"]].append(r["projE_r128_over_random"])
    print(f"  layer  projE(r=128)/random  (>1 => 向量偏向token主方向)")
    for L in sorted(agg):
        print(f"  {L:>5}  {sum(agg[L])/len(agg[L]):>8.2f}")

    # ============ 任务3: 沿 step 的熵演化 ============
    rows3=[]
    for exp_key in ("step_exp","step_exp_mid"):
        exp = cfg[exp_key]; exp_dir=os.path.join(cfg["vec_root"],exp)
        fs = sorted(glob.glob(exp_dir+"/vectors_step_*.pt"),
                    key=lambda p:int(re.search(r"step_(\d+)",p).group(1)))
        if not fs: continue
        # 采样最多 60 个 step 防止太慢
        idxs = list(range(0,len(fs), max(1,len(fs)//60)))
        for i in idxs:
            st=int(re.search(r"step_(\d+)",fs[i]).group(1))
            vecs=load_vecs(fs[i])
            for L,v in vecs.items():
                rows3.append({"model":"qwen3-4b","exp":exp,"layer":L,"step":st,
                              "norm_entropy":round(norm_entropy(rmsnorm(v,cfg["rms_eps"])@W.T),5),
                              "projE_r128":round(proj_energy(v,128),5),
                              "vec_norm":round(float(v.norm()),4)})
    with open(os.path.join(OUT,"entropy_over_steps.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows3[0].keys())); w.writeheader(); w.writerows(rows3)
    print(f"[task3] entropy_over_steps.csv  {len(rows3)} rows")

    # ---- 出图 ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # fig4: 能量分解 vs 层深 (多个 r)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for r in RANKS:
        a=defaultdict(list)
        for row in rows4: a[row["layer"]].append(row[f"projE_r{r}_over_random"])
        xs=sorted(a); ax.plot(xs,[sum(a[L])/len(a[L]) for L in xs],"o-",label=f"top-{r} dirs")
    ax.axhline(1.0,ls="--",color="gray",label="random vector")
    ax.set_xlabel("layer"); ax.set_ylabel("proj energy / random"); ax.set_yscale("log")
    ax.set_title("qwen3-4b: vector energy in W_U principal directions vs depth\n(>1 = aligned with token-discriminating subspace)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig4_wU_energy_vs_depth.png"),dpi=130)
    print("saved fig4")

    # fig5: 沿 step 的熵演化 (高层实验, 每层一条线)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,exp_key,ttl in [(axes[0],"step_exp","high layers 24-34"),
                           (axes[1],"step_exp_mid","mid layers 10-25")]:
        exp=cfg[exp_key]
        sub=[r for r in rows3 if r["exp"]==exp]
        bylayer=defaultdict(list)
        for r in sub: bylayer[r["layer"]].append((r["step"],r["norm_entropy"]))
        import matplotlib.cm as cm
        Ls=sorted(bylayer);
        for j,L in enumerate(Ls):
            pts=sorted(bylayer[L]);
            ax.plot([p[0] for p in pts],[p[1] for p in pts],
                    color=cm.viridis(j/max(1,len(Ls)-1)),label=f"L{L}" if L%3==0 or L==Ls[-1] else None)
        ax.set_xlabel("training step"); ax.set_ylabel("logit-lens norm entropy")
        ax.set_title(f"qwen3-4b {ttl}"); ax.grid(alpha=0.3); ax.legend(fontsize=7,ncol=2)
    fig.suptitle("Entropy evolution over training: high layers collapse (mode→token), mid layers stay flat",fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig5_entropy_over_steps.png"),dpi=130)
    print("saved fig5")

if __name__=="__main__":
    main()
