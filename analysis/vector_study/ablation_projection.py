#!/usr/bin/env python3
"""
实验3 (v2): 因果消融 —— norm-controlled projection ablation on OUTPUT distribution.

度量注入向量对"最终 next-token 分布"的因果影响 (ΔKL, 正是OPD loss优化的量), 并控制注入强度:
  对第L层向量 v:
    ΔKL_full    = KL( p(inject v) || p(no inject) )      注入原始 v
    ΔKL_ablated = KL( p(inject v_perp_scaled) || p(no) )  注入"投影掉W_U top-r主方向、再缩放回 ‖v‖"的向量
  retention = ΔKL_ablated / ΔKL_full

关键: v_perp 缩放回同样 norm, 所以比较的是"同等强度下, 方向在不在token(W_U)空间"。
预期:
  高层: 后续层≈恒等, 效应=W_U方向的直接logit改动 -> 去掉W_U方向后即使同norm也几乎无效 -> retention≈0 (token操纵)
  中低层: 效应经后续层非线性计算 -> 方向在不在W_U无所谓 -> retention≈1 (走计算通路)
retention vs 层深 从 ~1 掉到 ~0 = 因果证据: 高层作用全在token空间。
只需前向, 快。
"""
import os, glob, re, json, csv
import numpy as np
import torch

MP="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B"
VR="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors"
DATA="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
N_PROMPTS=24; MAX_TOK=200; RANK=256
PROBE=[1,3,7,10,13,16,20,24,27,30,33,34]

def load_layer_vec(L):
    best=None;bn=-1
    for d in sorted(glob.glob(VR+"/*")):
        if not os.path.isdir(d):continue
        fs=sorted(glob.glob(d+"/vectors_step_*.pt"),key=lambda p:int(re.search(r"step_(\d+)",p).group(1)))
        if not fs:continue
        dd=torch.load(fs[-1],map_location="cpu")
        if L in dd:
            t=dd[L];t=(t if torch.is_tensor(t) else t.get("vector")).float().reshape(-1)
            if t.norm()>bn:bn=float(t.norm());best=t
    return best

def main():
    import pandas as pd,numpy as np
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from safetensors import safe_open
    tok=AutoTokenizer.from_pretrained(MP,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(MP,torch_dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
    core=model.model
    idx=json.load(open(os.path.join(MP,"model.safetensors.index.json")))["weight_map"]
    k="model.embed_tokens.weight"
    with safe_open(os.path.join(MP,idx[k]),framework="pt",device="cpu") as f: W=f.get_tensor(k).float()
    print("SVD W_U..."); _,_,Vh=torch.svd_lowrank(W,q=RANK+16,niter=4); Vmain=Vh[:,:RANK]
    def ablate_scaled(v):
        c=Vmain.T@v; vperp=v-Vmain@c
        n=vperp.norm()
        if n<1e-9: return torch.zeros_like(v)
        return vperp*(v.norm()/n)          # 缩放回 ‖v‖
    def wU_frac(v):
        c=Vmain.T@v; return float((c@c)/(v@v+1e-12))

    df=pd.read_parquet(DATA)
    pcol=next((c for c in ["prompt","question","problem","query"] if c in df.columns),df.columns[0])
    prompts=[]
    for i in range(N_PROMPTS):
        row=df.iloc[i][pcol]
        if isinstance(row,(list,np.ndarray)):
            txt=tok.apply_chat_template(list(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        else:
            txt=tok.apply_chat_template([{"role":"user","content":str(row)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompts.append(txt)

    # 预取每个prompt的输入
    batches=[tok(p,return_tensors="pt",truncation=True,max_length=MAX_TOK).to("cuda:0") for p in prompts]
    # baseline logits (no inject), 缓存每prompt所有位置的 log-softmax
    def run(vec,L):
        cur={"v":vec}
        def hook(m,i,o):
            if cur["v"] is None:return o
            d=cur["v"].to(o[0].dtype).to(o[0].device)
            return (o[0]+d,)+o[1:] if isinstance(o,tuple) else o+d
        h=core.layers[L].register_forward_hook(hook) if vec is not None else None
        logps=[]
        with torch.no_grad():
            for b in batches:
                lg=model(**b).logits[0].float()          # [T,V]
                logps.append(torch.log_softmax(lg,-1).cpu())
        if h:h.remove()
        return logps

    # base 对每个探测层其实相同(不注入), 只需算一次
    print("baseline forward (no inject)...")
    base_logp=run(None,0)
    def meanKL(inj_logp):
        # KL(p_inject || p_base) 逐位置平均
        ks=[]
        for pb,pi in zip(base_logp,inj_logp):
            p=pi.exp()
            kl=(p*(pi-pb)).sum(-1)          # [T]
            ks.append(kl.mean().item())
        return float(np.mean(ks))

    rows=[]
    for L in PROBE:
        v=load_layer_vec(L)
        if v is None: print(f"[skip]L{L}");continue
        vab=ablate_scaled(v); fr=wU_frac(v)
        kl_full=meanKL(run(v,L))
        kl_abl =meanKL(run(vab,L))
        ret=kl_abl/kl_full if kl_full>1e-9 else float("nan")
        rows.append(dict(layer=L,vec_norm=round(float(v.norm()),3),wU_frac=round(fr,4),
                         dKL_full=round(kl_full,5),dKL_ablated=round(kl_abl,5),retention=round(ret,4)))
        print(f"L{L:>2} wU_frac={fr:.3f} dKL_full={kl_full:.4f} dKL_abl(scaled)={kl_abl:.4f} retention={ret:.3f}")

    with open(os.path.join(OUT,"ablation_dkl.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote ablation_dkl.csv")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ls=[r["layer"] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    axes[0].plot(Ls,[r["dKL_full"] for r in rows],"o-",color="crimson",label="full v")
    axes[0].plot(Ls,[r["dKL_ablated"] for r in rows],"o-",color="orange",label="W_U-ablated (same norm)")
    axes[0].set_xlabel("injected layer"); axes[0].set_ylabel("ΔKL on final output dist")
    axes[0].set_title("causal effect on output: full vs token-ablated (norm-matched)")
    axes[0].legend();axes[0].grid(alpha=0.3)
    axes[1].plot(Ls,[r["retention"] for r in rows],"o-",color="navy")
    axes[1].axhline(1,ls="--",color="gray",label="direction irrelevant (computation)")
    axes[1].axhline(0,ls=":",color="k")
    axes[1].set_xlabel("injected layer"); axes[1].set_ylabel("retention = ΔKL_ablated / ΔKL_full")
    axes[1].set_title("high layers → retention↓ (effect needs token direction)\nmid/low → retention≈1 (effect via computation)")
    axes[1].legend();axes[1].grid(alpha=0.3)
    fig.suptitle("Projection ablation (norm-controlled): is the vector's output effect carried by the W_U/token direction?",fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig13_projection_ablation.png"),dpi=130); print("saved fig13")

if __name__=="__main__": main()
