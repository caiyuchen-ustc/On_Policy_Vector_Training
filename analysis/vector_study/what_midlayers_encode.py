#!/usr/bin/env python3
"""
任务6 (A): 最大激活分析 —— 中低层向量方向代表什么"概念"。
  对第 L 层, 真实语料每个 token 的 residual h_t, 算 align = cos(h_t, v_L)。
  找 align 最高/最低的 token 及其左右上下文, 并按 token 文本聚合出现频次。
  这直接回答"这个方向在什么语境下发火"。

任务7 (B): 因果 push-through —— 中低层"模式"的真实下游效应 (logit lens 对低层无效的修正)。
  取真实序列最后位置的 h_L, 构造 h_L + s*v_L (s 用该层实际 vec_norm 或标定尺度),
  手动跑完剩余层 + norm + lm_head, 得到扰动后 logits, 对比原 logits,
  看 Δlogit 最大的 token。这不再假设后续层恒等, 是真实的因果效应。
  同时对比"直接 logit-lens"预测的 token, 看两者是否一致:
    - 高层: push-through 和 logit-lens 一致 (后续层≈恒等, 就是token操纵)
    - 中低层: 若 logit-lens 是乱码但 push-through 是有语义token -> 证明是"经过计算"的模式
"""
import os, glob, re, csv, json
from collections import Counter, defaultdict
import torch

MODELS={
    "qwen3-4b": dict(
        path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B",
        vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
        probe=(1,3,7,10,15,20,24,27,30,33,34),
    ),
    "deepseek7b": dict(
        path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B",
        vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/ifevalg_opd/trainable_vectors",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/ifevalg/ifevalg_train.parquet",
        probe=(1,3,7,10,14,17,20,23,25,26,27),
    ),
}
_TAG=os.environ.get("MODEL_TAG","qwen3-4b")
MODEL_PATH=MODELS[_TAG]["path"]
VEC_ROOT=MODELS[_TAG]["vec_root"]
DATA=MODELS[_TAG]["data"]
OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
os.makedirs(OUT,exist_ok=True)

N_PROMPTS=24
MAX_TOK=200
TOPN=40           # 每层保留 top/bottom N 激活 token
CTX=6             # 上下文窗口
PUSH_SCALE=8.0    # push-through 时 v 的注入强度(相对 normalize 后), 设大以放大可读效应


def load_final_vectors():
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
                best[L]=(t,float(t.norm()),os.path.basename(exp_dir))
    return {L:t for L,(t,n,e) in best.items()}


def main():
    import pandas as pd, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vecs=load_final_vectors()
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16,
                                               device_map="cuda:0",trust_remote_code=True).eval()
    core=model.model; nlayer=len(core.layers); Hd=model.config.hidden_size
    # 归一化后的方向 (device)
    dirs={L:(v/ (v.norm()+1e-9)).to("cuda:0").bfloat16() for L,v in vecs.items()}

    df=pd.read_parquet(DATA)
    pcol=next((c for c in ["prompt","question","problem","query"] if c in df.columns), df.columns[0])
    prompts=[]
    for i in range(min(N_PROMPTS,len(df))):
        row=df.iloc[i][pcol]
        if isinstance(row,(list,np.ndarray)):
            txt=tok.apply_chat_template(list(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        else:
            txt=tok.apply_chat_template([{"role":"user","content":str(row)}],tokenize=False,
                                        add_generation_prompt=True,enable_thinking=False)
        prompts.append(txt)

    # 收集每层 hidden (含 token id 和 序列位置, 用于回溯上下文)
    layer_h=defaultdict(list)     # L -> list of [T,H] tensors (cpu f32)
    seq_ids=[]                    # 每个 prompt 的 token id list
    handles=[]
    def mk(L):
        def hook(m,i,o):
            h=o[0] if isinstance(o,tuple) else o
            layer_h[L].append(h.detach()[0].float().cpu())   # [T,H], B=1
        return hook
    for L,layer in enumerate(core.layers):
        handles.append(layer.register_forward_hook(mk(L)))
    with torch.no_grad():
        for p in prompts:
            ids=tok(p,return_tensors="pt",truncation=True,max_length=MAX_TOK).to("cuda:0")
            seq_ids.append(ids["input_ids"][0].cpu().tolist())
            model(**ids)
    for h in handles: h.remove()

    # ---------- 任务6 (A): 最大激活 ----------
    actrows=[]
    for L in sorted(vecs):
        v=dirs[L].float().cpu()
        # 拼所有 token: 记录 (score, prompt_idx, pos)
        recs=[]
        for pi,h in enumerate(layer_h[L]):
            hn=h/ (h.norm(dim=-1,keepdim=True)+1e-9)
            sc=(hn@v)   # [T]
            for pos in range(h.shape[0]):
                recs.append((float(sc[pos]),pi,pos))
        recs.sort(reverse=True)
        top=recs[:TOPN]; bot=recs[-TOPN:]
        def tokstr(pi,pos):
            ids=seq_ids[pi]
            cur=tok.decode([ids[pos]]).replace("\n","\\n")
            left=tok.decode(ids[max(0,pos-CTX):pos]).replace("\n","\\n")
            right=tok.decode(ids[pos+1:pos+1+CTX]).replace("\n","\\n")
            return cur,f"{left}⟪{cur}⟫{right}"
        top_toks=Counter(); bot_toks=Counter()
        top_ctx=[];
        for sc,pi,pos in top:
            c,ctx=tokstr(pi,pos); top_toks[c.strip()]+=1; top_ctx.append((round(sc,3),ctx))
        for sc,pi,pos in bot:
            c,_=tokstr(pi,pos); bot_toks[c.strip()]+=1
        actrows.append({"layer":L,
            "top_tokens":" | ".join(f"{t}×{n}" for t,n in top_toks.most_common(15)),
            "bottom_tokens":" | ".join(f"{t}×{n}" for t,n in bot_toks.most_common(10)),
            "max_cos":round(top[0][0],4),"min_cos":round(bot[-1][0],4),
            "top_ctx_examples":" ||| ".join(c for _,c in top_ctx[:5])})
    with open(os.path.join(OUT,f"max_activation_{_TAG}.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(actrows[0].keys())); w.writeheader(); w.writerows(actrows)
    # 人类可读
    with open(os.path.join(OUT,f"max_activation_readable_{_TAG}.txt"),"w") as fh:
        for r in actrows:
            fh.write(f"\n===== Layer {r['layer']}  (max_cos={r['max_cos']}, min_cos={r['min_cos']}) =====\n")
            fh.write(f"  TOP activating tokens: {r['top_tokens']}\n")
            fh.write(f"  BOTTOM (anti) tokens : {r['bottom_tokens']}\n")
            fh.write(f"  contexts:\n")
            for c in r['top_ctx_examples'].split(" ||| "):
                fh.write(f"    {c}\n")
    print(f"[task6:{_TAG}] wrote max_activation")

    # ---------- 任务7 (B): 因果 push-through ----------
    # 用最后一层前的 final norm + lm_head 手动前向剩余层
    norm=core.norm; lm_head=model.lm_head
    def push(L, s):
        """取所有prompt最后位置的 h_L, +s*dir, 跑完剩余层, 返回平均 Δlogit 的 top token。"""
        base_logits=[]; steer_logits=[]
        with torch.no_grad():
            for pi,h in enumerate(layer_h[L]):
                hpos=h[-1:].to("cuda:0").bfloat16()      # [1,H] 最后位置
                # 需要重建后续层输入: 用 hidden 直接喂后续 decoder 层 (近似, 忽略attention对历史的依赖:
                # 单token最后位置, 后续层的self-attn需要KV. 简化: 只测最后位置对自身的前向,
                # 用 position_ids 和单步. Qwen3 层需要 position_embeddings, 这里退化为
                # 仅测 MLP+norm 通路会不准, 所以改为完整重跑: 见下方 full_rerun)
                pass
        return None

    # 完整重跑法: 注入在第L层输出, 用 forward hook 改写, 再取整句最后位置 logits
    def full_rerun(L, s):
        vdir=dirs[L]
        def add_hook(m,i,o):
            if isinstance(o,tuple):
                return (o[0]+s*vdir.to(o[0].dtype),)+o[1:]
            return o+s*vdir.to(o.dtype)
        base=[]; steer=[]
        with torch.no_grad():
            for p in prompts[:12]:
                ids=tok(p,return_tensors="pt",truncation=True,max_length=MAX_TOK).to("cuda:0")
                lo=model(**ids).logits[0,-1].float().cpu(); base.append(lo)
                hd=core.layers[L].register_forward_hook(add_hook)
                lo2=model(**ids).logits[0,-1].float().cpu(); steer.append(lo2)
                hd.remove()
        base=torch.stack(base).mean(0); steer=torch.stack(steer).mean(0)
        dl=steer-base
        # 直接 logit-lens 预测(对照)
        v=vecs[L]; ll=(v/torch.sqrt((v*v).mean()+1e-6))@ (core.embed_tokens.weight.float().cpu().T)
        top_push=torch.topk(dl,12).indices.tolist()
        top_ll=torch.topk(ll,12).indices.tolist()
        overlap=len(set(top_push)&set(top_ll))
        return ([tok.decode([i]).replace("\n","\\n") for i in top_push],
                [tok.decode([i]).replace("\n","\\n") for i in top_ll], overlap)

    pushrows=[]
    probe_layers=[L for L in sorted(vecs) if L in set(MODELS[_TAG]["probe"])]
    for L in probe_layers:
        push_toks,ll_toks,ov=full_rerun(L, PUSH_SCALE)
        pushrows.append({"layer":L,"pushthrough_top":" | ".join(push_toks),
                         "logitlens_top":" | ".join(ll_toks),"overlap_top12":ov})
        print(f"L{L}: push-through Δlogit top -> {' | '.join(push_toks[:8])}")
    with open(os.path.join(OUT,f"pushthrough_{_TAG}.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(pushrows[0].keys())); w.writeheader(); w.writerows(pushrows)
    print(f"[task7:{_TAG}] wrote pushthrough")

if __name__=="__main__":
    main()
