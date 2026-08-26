#!/usr/bin/env python3
"""
补充1: 按 token 类型统计每层向量对齐的 token, 画"中低层偏结构 vs 高层偏反思词"的热力图。
数据源: 直接复用 what_midlayers_encode.py 的前向 (重新收集 hidden), 对每层取 top-200 激活 token,
分类到几大类, 算各类占比 vs 层深。

类别:
  reflection : 反思/转折推理词  wait/but/hmm/alternatively/perhaps/maybe/however/okay/alright/so + 中文 但/或许/然而/嗯
  structural : 结构/角色/控制   <think></think> assistant im_start im_end \n reason boxed final step your
  math       : 数学符号/概念     单字符符号 = + - ^ _ { } \ ( [ 及 frac/sqrt/mathbb/matrix/metric/fiber/smooth/algebra/topology/bounded/canonical/product...
  other      : 其余 (自然语言实词/子词碎片)
"""
import os, glob, re, csv
from collections import defaultdict
import torch

MODELS={
 "qwen3-4b":dict(path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B",
   vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors",
   data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"),
 "deepseek7b":dict(path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B",
   vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/ifevalg_opd/trainable_vectors",
   data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/ifevalg/ifevalg_train.parquet"),
}
_TAG=os.environ.get("MODEL_TAG","qwen3-4b")
MODEL_PATH=MODELS[_TAG]["path"]; VEC_ROOT=MODELS[_TAG]["vec_root"]; DATA=MODELS[_TAG]["data"]
OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
N_PROMPTS=24; MAX_TOK=200; TOPK=200

REFLECT={"wait","but","hmm","alternatively","perhaps","maybe","however","okay","ok","alright",
         "so","actually","但","或许","然而","嗯","那么","不过","等待","也许"}
STRUCT_SUB=["think","assistant","im_start","im_end","reason","boxed","final","step","your","user","system"]
STRUCT_EXACT={"\\n","\\n\\n",r"\n",r"\n\n","\n","\n\n"}
MATH_SUB=["frac","sqrt","mathbb","matrix","metric","fiber","smooth","algebra","topology","bounded",
          "canonical","product","divisor","integral","theorem","lemma","phi","tau","sigma","infty","sum","prod","lim"]
MATH_SYM=set(list("=+-^_{}\\()[]$*/|<>&"))


def classify(t):
    s=t.strip(); low=s.lower()
    if low in REFLECT: return "reflection"
    if s in STRUCT_EXACT or s=="" or all(c=='\n' for c in t) or t in ("\\n","\\n\\n"): return "structural"
    if any(k in low for k in STRUCT_SUB): return "structural"
    if s in MATH_SYM or (len(s)<=2 and any(c in MATH_SYM for c in s)): return "math"
    if any(k in low for k in MATH_SUB): return "math"
    if low in REFLECT: return "reflection"
    return "other"


def load_final_vectors():
    best={}
    for exp_dir in sorted(glob.glob(os.path.join(VEC_ROOT,"*"))):
        if not os.path.isdir(exp_dir): continue
        fs=sorted(glob.glob(exp_dir+"/vectors_step_*.pt"),key=lambda p:int(re.search(r"step_(\d+)",p).group(1)))
        if not fs: continue
        d=torch.load(fs[-1],map_location="cpu")
        for k,v in d.items():
            t=v if torch.is_tensor(v) else v.get("vector")
            if t is None: continue
            t=t.float().reshape(-1); L=int(k)
            if L not in best or t.norm()>best[L][1]: best[L]=(t,float(t.norm()))
    return {L:t for L,(t,n) in best.items()}


def main():
    import pandas as pd, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vecs=load_final_vectors()
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16,
                                               device_map="cuda:0",trust_remote_code=True).eval()
    core=model.model
    df=pd.read_parquet(DATA)
    pcol=next((c for c in ["prompt","question","problem","query"] if c in df.columns),df.columns[0])
    prompts=[]
    for i in range(min(N_PROMPTS,len(df))):
        row=df.iloc[i][pcol]
        if isinstance(row,(list,np.ndarray)):
            txt=tok.apply_chat_template(list(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        else:
            txt=tok.apply_chat_template([{"role":"user","content":str(row)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompts.append(txt)
    layer_h=defaultdict(list); seq_ids=[]; handles=[]
    def mk(L):
        def hook(m,i,o):
            h=o[0] if isinstance(o,tuple) else o
            layer_h[L].append(h.detach()[0].float().cpu())
        return hook
    for L,layer in enumerate(core.layers): handles.append(layer.register_forward_hook(mk(L)))
    with torch.no_grad():
        for p in prompts:
            ids=tok(p,return_tensors="pt",truncation=True,max_length=MAX_TOK).to("cuda:0")
            seq_ids.append(ids["input_ids"][0].cpu().tolist()); model(**ids)
    for h in handles: h.remove()

    cats=["reflection","structural","math","other"]
    rows=[]
    for L in sorted(vecs):
        v=(vecs[L]/ (vecs[L].norm()+1e-9))
        recs=[]
        for pi,h in enumerate(layer_h[L]):
            hn=h/(h.norm(dim=-1,keepdim=True)+1e-9); sc=hn@v
            for pos in range(h.shape[0]): recs.append((float(sc[pos]),pi,pos))
        recs.sort(reverse=True)
        cnt=defaultdict(int)
        for sc,pi,pos in recs[:TOPK]:
            t=tok.decode([seq_ids[pi][pos]])
            cnt[classify(t)]+=1
        tot=sum(cnt.values())
        rows.append({"layer":L,**{c:round(cnt[c]/tot,4) for c in cats}})
    with open(os.path.join(OUT,f"token_type_by_layer_{_TAG}.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=["layer"]+cats); w.writeheader(); w.writerows(rows)
    print("wrote token_type_by_layer.csv")
    for r in rows:
        print(f"L{r['layer']:>2}  refl={r['reflection']:.2f} struct={r['structural']:.2f} math={r['math']:.2f} other={r['other']:.2f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    Ls=[r["layer"] for r in rows]
    M=np.array([[r[c] for r in rows] for c in cats])  # [4, nL]
    fig,ax=plt.subplots(figsize=(13,3.2))
    im=ax.imshow(M,aspect="auto",cmap="viridis",vmin=0,vmax=0.7)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels(Ls,fontsize=7)
    ax.set_xlabel("layer"); ax.set_title("Top-activating token type composition by layer (Qwen3-4B single vectors)")
    fig.colorbar(im,ax=ax,fraction=0.02); fig.tight_layout()
    fig.savefig(os.path.join(FIG,f"fig7_token_type_heatmap_{_TAG}.png"),dpi=130); print("saved fig7")
    # 折线: reflection 占比 vs 层深
    fig,ax=plt.subplots(figsize=(7.5,4))
    for c,col in [("reflection","crimson"),("structural","navy"),("math","green")]:
        ax.plot(Ls,[r[c] for r in rows],"o-",label=c,color=col)
    ax.set_xlabel("layer"); ax.set_ylabel("fraction of top-200 activating tokens")
    ax.set_title("mid/low layers → structural; high layers → reflection words")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIG,f"fig8_tokentype_lines_{_TAG}.png"),dpi=130); print("saved fig8")

if __name__=="__main__": main()
