#!/usr/bin/env python3
"""
任务5: 训练向量 vs vLLM/HF 推理中真实 hidden_state 的空间关系。

思路: 单向量 v_L 被加到第 L 层输出的 residual stream h_L 上。要回答"v 落在 h 的主空间还是尾空间":
  1) 跑真实前向, 抓每层 residual stream h_L (跨很多 token 位置), 得到样本矩阵 X_L [N, H]
  2) 对 X_L 做 PCA (中心化后 SVD), 得主成分 PC_k 和解释方差 λ_k
  3) 度量 v_L 与该 hidden 分布的关系:
     - align_meandir : cos(v, mean(h))           是否顺着 hidden 的整体偏置方向
     - energy_in_topk: v 在前 k 个 PC 上的能量占比 (k=1,5,20,64) vs 随机基线 k/H
     - pc_profile    : v 在每个 PC 上投影^2 / λ_k  ->  “v 沿高方差方向还是低方差方向”
     - inject_dnorm  : ‖v‖ / mean‖h‖              注入扰动相对幅度
     - cos_to_pc1    : cos(v, PC1)

  关键判据:
     若 v 落在 hidden 的低方差"尾空间"(前k能量≈随机甚至更低) -> 操控一个原本不活跃的稀有方向 (高层可能这样)
     若 v 落在高方差"主空间"(前k能量>>随机) -> 顺着模型本来就在用的表征方向 (中低层的"模式")
"""
import os, glob, re, csv, json, argparse
import torch
import numpy as np

# 两个模型可选, 用 MODEL_TAG 环境变量或 --model 切换
MODELS={
    "qwen3-4b": dict(
        path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B",
        vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
    ),
    "deepseek7b": dict(
        path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B",
        vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/ifevalg_opd/trainable_vectors",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/ifevalg/ifevalg_train.parquet",
    ),
}
_TAG=os.environ.get("MODEL_TAG","qwen3-4b")
MODEL_PATH=MODELS[_TAG]["path"]
VEC_ROOT=MODELS[_TAG]["vec_root"]
DATA=MODELS[_TAG]["data"]
OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
os.makedirs(OUT,exist_ok=True); os.makedirs(FIG,exist_ok=True)


def load_final_vectors():
    """收集所有实验里每层的最终向量。多个实验含同层时, 取 norm 最大(训得最充分)的一个。"""
    best={}
    for exp_dir in sorted(glob.glob(os.path.join(VEC_ROOT,"*"))):
        if not os.path.isdir(exp_dir): continue
        fs=sorted(glob.glob(exp_dir+"/vectors_step_*.pt"),
                  key=lambda p:int(re.search(r"step_(\d+)",p).group(1)))
        if not fs: continue
        d=torch.load(fs[-1],map_location="cpu")
        for k,v in d.items():
            t=v if torch.is_tensor(v) else v.get("vector")
            if t is None: continue
            t=t.float().reshape(-1); L=int(k)
            if L not in best or t.norm()>best[L][1]:
                best[L]=(t, float(t.norm()), os.path.basename(exp_dir))
    return {L:(t,exp) for L,(t,n,exp) in best.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n_prompts",type=int,default=32)
    ap.add_argument("--max_tokens",type=int,default=256, help="每个prompt截断长度")
    args=ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vecs=load_final_vectors()
    layers_with_vec=sorted(vecs)
    print("layers with trained vector:", layers_with_vec)

    tok=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16,
                                               device_map="cuda:0",trust_remote_code=True)
    model.eval()

    # 取 prompt
    df=pd.read_parquet(DATA)
    # 找到 prompt 列
    pcol=None
    for c in ["prompt","question","problem","query"]:
        if c in df.columns: pcol=c; break
    if pcol is None:
        # prompt 可能是 chat list
        pcol=df.columns[0]
    prompts=[]
    for i in range(min(args.n_prompts,len(df))):
        row=df.iloc[i][pcol]
        if isinstance(row,(list,np.ndarray)):
            txt=tok.apply_chat_template(list(row),tokenize=False,add_generation_prompt=True,
                                        enable_thinking=False)
        elif isinstance(row,str):
            txt=tok.apply_chat_template([{"role":"user","content":row}],tokenize=False,
                                        add_generation_prompt=True,enable_thinking=False)
        else:
            txt=str(row)
        prompts.append(txt)
    print(f"prompt col={pcol}, n={len(prompts)}")

    # hook: 抓每个 decoder layer 输出 (residual stream)
    core=model.model  # Qwen3Model
    nlayer=len(core.layers)
    collected={L:[] for L in range(nlayer)}
    handles=[]
    def mk(L):
        def hook(m,inp,out):
            h=out[0] if isinstance(out,tuple) else out   # [B,T,H]
            collected[L].append(h.detach().float().reshape(-1,h.shape[-1]).cpu())
        return hook
    for L,layer in enumerate(core.layers):
        handles.append(layer.register_forward_hook(mk(L)))

    with torch.no_grad():
        for p in prompts:
            ids=tok(p,return_tensors="pt",truncation=True,max_length=args.max_tokens).to("cuda:0")
            model(**ids)
    for h in handles: h.remove()

    # 逐层分析
    rows=[]
    pc_profiles={}
    H=model.config.hidden_size
    for L in range(nlayer):
        X=torch.cat(collected[L],0)   # [N,H]
        N=X.shape[0]
        mean=X.mean(0)
        Xc=X-mean
        # PCA via SVD (随机投影降算力: 用 torch.pca_lowrank)
        q=min(128,H,N-1)
        Uu,Ss,Vv=torch.pca_lowrank(Xc, q=q, niter=3)   # Vv:[H,q] 主成分, Ss 奇异值
        lam=(Ss**2)/(N-1)                              # 各PC方差
        var_total=(Xc*Xc).sum()/(N-1)                  # 总方差(近似)
        rec={"layer":L,"N":N,"has_vec":int(L in vecs),
             "hid_mean_norm":round(float(mean.norm()),3),
             "hid_rms":round(float(X.pow(2).mean().sqrt()),3)}
        if L in vecs:
            v,exp=vecs[L]; vn=v/ (v.norm()+1e-9)
            rec["exp"]=exp; rec["vec_norm"]=round(float(v.norm()),4)
            rec["cos_meandir"]=round(float(torch.dot(vn,mean/ (mean.norm()+1e-9))),4)
            rec["inject_rel"]=round(float(v.norm()/X.norm(dim=-1).mean()),4)
            # v 在前 k 个 PC 的能量占比
            proj=(Vv.T@vn)     # [q]  单位向量在各PC的坐标
            cum=torch.cumsum(proj**2,0)
            for k in [1,5,20,64]:
                if k<=q:
                    rec[f"E_top{k}"]=round(float(cum[k-1]),4)
                    rec[f"E_top{k}_over_rand"]=round(float(cum[k-1])/(k/H),3)
            rec["cos_pc1"]=round(float(abs(proj[0])),4)
            # v 落点的“方差加权重心”: sum(proj_k^2 * rank_k)/sum(proj_k^2)  低=偏主成分, 高=偏尾
            ranks=torch.arange(1,q+1,dtype=torch.float32)
            rec["pc_centroid_rank"]=round(float((proj**2*ranks).sum()/(proj**2).sum()),2)
            # 对照: 随机向量的重心 ~ (q+1)/2
            rec["pc_centroid_rand"]=round((q+1)/2,1)
            pc_profiles[L]=(proj.numpy()**2, lam.numpy())
        rows.append(rec)
        print(f"L{L:>2} N={N} "+(f"vec_norm={rec.get('vec_norm')} cos_meandir={rec.get('cos_meandir')} "
              f"E_top20/rand={rec.get('E_top20_over_rand')} centroid_rank={rec.get('pc_centroid_rank')}/{rec.get('pc_centroid_rand')}" if L in vecs else "(no vec)"))

    # 写 CSV
    keys=sorted({k for r in rows for k in r})
    with open(os.path.join(OUT,f"hidden_subspace_{_TAG}.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=["layer","has_vec","exp","N","vec_norm","hid_mean_norm","hid_rms",
            "cos_meandir","inject_rel","cos_pc1","E_top1","E_top1_over_rand","E_top5","E_top5_over_rand",
            "E_top20","E_top20_over_rand","E_top64","E_top64_over_rand","pc_centroid_rank","pc_centroid_rand"])
        w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in w.fieldnames})
    print("wrote hidden_subspace.csv")

    # 图: E_top20/rand vs layer  和  pc_centroid_rank vs layer
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vl=[r for r in rows if r["has_vec"]]
    vl.sort(key=lambda r:r["layer"])
    Ls=[r["layer"] for r in vl]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].plot(Ls,[r["E_top20_over_rand"] for r in vl],"o-",color="crimson")
    axes[0].axhline(1,ls="--",color="gray",label="random vector")
    axes[0].set_xlabel("layer");axes[0].set_ylabel("energy in hidden top-20 PC / random")
    axes[0].set_title("vector energy in hidden PRINCIPAL subspace\n(>1 aligns with high-variance 'mode' directions)")
    axes[0].legend();axes[0].grid(alpha=0.3)
    axes[1].plot(Ls,[r["pc_centroid_rank"] for r in vl],"o-",color="navy",label="trained vector")
    axes[1].plot(Ls,[r["pc_centroid_rand"] for r in vl],"--",color="gray",label="random vector")
    axes[1].set_xlabel("layer");axes[1].set_ylabel("PC centroid rank (low=principal, high=tail)")
    axes[1].set_title("where in hidden spectrum does v live?\n(low rank = main space, high rank = unimportant space)")
    axes[1].legend();axes[1].grid(alpha=0.3)
    fig.tight_layout();fig.savefig(os.path.join(FIG,f"fig6_vector_vs_hidden_subspace_{_TAG}.png"),dpi=130)
    print("saved fig6")

if __name__=="__main__":
    main()
