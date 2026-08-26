#!/usr/bin/env python3
"""
App. C.6: 跨-run 典范子空间一致性。
同一层在不同实验(不同层范围/lr/seed)独立学到的最终 single 向量:
  - mean pairwise cosine  : 方向是否收敛到同一"典范方向"(vs 随机≈0)
  - participation ratio(PR): 堆叠SVD的有效维度, 反映该层RL增益占几维子空间
支撑正文 X.4 "每层1-2维典范子空间"论断。
"""
import os, glob, re, itertools, csv
from collections import defaultdict
import torch

VR="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors"
OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"

def main():
    per=defaultdict(list)
    for d in sorted(glob.glob(VR+"/*")):
        if not os.path.isdir(d): continue
        fs=sorted(glob.glob(d+"/vectors_step_*.pt"),key=lambda p:int(re.search(r"step_(\d+)",p).group(1)))
        if not fs: continue
        dd=torch.load(fs[-1],map_location="cpu")
        for k,v in dd.items():
            t=v if torch.is_tensor(v) else v.get("vector")
            if t is None: continue
            t=t.float().reshape(-1)
            per[int(k)].append(t/t.norm())
    def cos(a,b): return float(torch.dot(a,b))
    rows=[]
    for L in sorted(per):
        vs=per[L]
        if len(vs)<2: continue
        cs=[cos(a,b) for a,b in itertools.combinations(vs,2)]
        mc=sum(cs)/len(cs)
        pr=None
        if len(vs)>=3:
            M=torch.stack(vs); s=torch.linalg.svdvals(M); e=(s**2); e=e/e.sum()
            pr=float((e.sum()**2)/(e**2).sum())
        rows.append({"layer":L,"n_runs":len(vs),"mean_pairwise_cos":round(mc,4),
                     "participation_ratio":round(pr,3) if pr else ""})
    with open(os.path.join(OUT,"crossrun_consistency.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R=[r for r in rows if r["mean_pairwise_cos"]!=""]
    Ls=[r["layer"] for r in R]
    fig,ax=plt.subplots(1,2,figsize=(13,4.4))
    ax[0].plot(Ls,[r["mean_pairwise_cos"] for r in R],"o-",color="teal")
    ax[0].axhline(0,ls=":",color="k",label="random vectors ≈ 0")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("mean pairwise cosine (independent runs)")
    ax[0].set_title("Cross-run direction agreement\n(high = canonical direction, not random solution)")
    ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_ylim(-0.05,1)
    R2=[r for r in rows if r["participation_ratio"]!=""]
    ax[1].plot([r["layer"] for r in R2],[r["participation_ratio"] for r in R2],"o-",color="purple")
    ax[1].axhline(1,ls="--",color="gray",label="1D (a single line)")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("participation ratio of stacked runs")
    ax[1].set_title("Effective dimensionality per layer\n(≈1-2D canonical subspace)")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_ylim(0,4)
    fig.suptitle("The per-layer RL gain occupies a low-dimensional, canonical subspace",fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"figC_crossrun_consistency.png"),dpi=130)
    print("saved figC_crossrun_consistency.png")

if __name__=="__main__": main()
