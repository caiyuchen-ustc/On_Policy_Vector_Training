#!/usr/bin/env python3
"""双模型对比总图: qwen3-4b (数学OPD) vs deepseek7b (IF OPD)。
把 logit-lens熵、W_U能量、hidden子空间能量 三个指标按 归一化层深(depth/nlayer) 画在一起对比。"""
import os,csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
NLAYER={"qwen3-4b":36,"deepseek7b":28}
BASE_NE={"qwen3-4b":0.9498,"deepseek7b":0.9594}

def rd(p):
    with open(p) as f: return list(csv.DictReader(f))
def lm(rows,key,mode=None):
    a=defaultdict(list)
    for r in rows:
        if mode and r.get("mode")!=mode: continue
        try: a[int(r["layer"])].append(float(r[key]))
        except: pass
    xs=sorted(a); return xs,[sum(a[L])/len(a[L]) for L in xs]

fig,axes=plt.subplots(1,3,figsize=(16,4.6))
COL={"qwen3-4b":"crimson","deepseek7b":"navy"}
LAB={"qwen3-4b":"qwen3-4b (math OPD)","deepseek7b":"deepseek7b (IF OPD)"}

# panel1: logit-lens entropy vs normalized depth
for m in NLAYER:
    rows=rd(os.path.join(OUT,f"logitlens_{m}.csv"))
    xs,ys=lm(rows,"norm_entropy",mode="normed")
    nl=NLAYER[m]
    axes[0].plot([x/(nl-1) for x in xs],ys,"o-",color=COL[m],label=LAB[m],ms=4)
    axes[0].axhline(BASE_NE[m],ls=":",color=COL[m],alpha=0.5)
axes[0].set_xlabel("normalized depth (layer/L)"); axes[0].set_ylabel("logit-lens norm entropy")
axes[0].set_title("(1) logit-lens sharpness\nlow at top = token collapse"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(0,1.02)

# panel2: W_U principal energy vs depth
for m in NLAYER:
    p=os.path.join(OUT,f"wU_energy_{m}.csv") if m!="qwen3-4b" else os.path.join(OUT,"wU_energy.csv")
    rows=rd(p); nl=NLAYER[m]
    xs,ys=lm(rows,"projE_r128_over_random")
    axes[1].plot([x/(nl-1) for x in xs],ys,"o-",color=COL[m],label=LAB[m],ms=4)
axes[1].axhline(1,ls="--",color="gray",label="random")
axes[1].set_xlabel("normalized depth"); axes[1].set_ylabel("energy in W_U top-128 / random")
axes[1].set_title("(2) alignment w/ unembedding\nhigh at top = token space"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# panel3: hidden top-20 PC energy vs depth
for m in NLAYER:
    p=os.path.join(OUT,f"hidden_subspace_{m}.csv") if m!="qwen3-4b" else os.path.join(OUT,"hidden_subspace.csv")
    rows=[r for r in rd(p) if r.get("has_vec")=="1" and r.get("E_top20_over_rand")]
    nl=NLAYER[m]
    xs=[int(r["layer"]) for r in rows]; ys=[float(r["E_top20_over_rand"]) for r in rows]
    axes[2].plot([x/(nl-1) for x in xs],ys,"o-",color=COL[m],label=LAB[m],ms=4)
axes[2].axhline(1,ls="--",color="gray",label="random")
axes[2].set_xlabel("normalized depth"); axes[2].set_ylabel("energy in hidden top-20 PC / random")
axes[2].set_title("(3) alignment w/ hidden main space"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

fig.suptitle("Two-model comparison: both show high-layer vectors drift into token/output subspace (qwen3-4b more extreme, has more top layers)",fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig10_two_model_compare.png"),dpi=130)
print("saved fig10")
