#!/usr/bin/env python3
"""
实验2: per-position 梯度冲突分析 —— 解释为什么高层单向量 norm 失控/难收敛。

单向量被加到某层所有 token 位置。理想更新方向是"让每个位置的输出更接近 teacher"。
第 L 层位置 t 的理想 residual 更新方向可近似为 reverse-KL 对 h_t 的梯度:
    g_t ≈ J_L^T · W_U^T · (p_teacher,t − p_student,t)
其中 J_L 是第L层到输出的 Jacobian。高层 J_L≈I(后续层近恒等), 故:
    高层  g_t ≈ W_U^T (p_teacher − p_student)_t     —— 直接由 token 分布差决定
    低层  J_L 是复杂非线性, g_t 被"打散"到各种方向

关键度量: 同一层内, 不同位置 g_t 的方向一致性
    mean pairwise cos(g_i, g_j)。
    - 一致性高 -> 一个全局向量能同时满足所有位置 -> 好训、norm稳定
    - 一致性低(甚至负) -> 各位置要求互相冲突 -> 全局向量被拉扯 -> norm失控/震荡

高层 g_t 由 (p_teacher−p_student) 主导, 不同位置该预测的 token 完全不同(有的该出数字,有的该出Wait),
故 g_t 高度冲突。低层 g_t 经 Jacobian 打散到"模式"方向, 冲突小。
我们用两种近似算 g_t 并比较跨层的位置间一致性:
  (a) direct: g_t = W_U^T (p_teacher,t − p_student,t)   —— 高层的真实梯度近似
  (b) 真实反传太贵, 用 (a) 作为"若后续层恒等"的理想梯度, 直接对比各层 hidden 下该梯度的位置一致性。
另外报告一个不依赖 teacher 的量: 各层 hidden 的"每位置最优 logit 增量方向" W_U^T(onehot(argmax_next) − p) 的位置间一致性,
反映"该层若要直接改输出, 各位置需求有多冲突"。
"""
import os, glob, re, csv, json
import torch

MODEL_PATH="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B"
TEACHER="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B-Non-Thinking-RL-Math-Step500"
DATA="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
N_PROMPTS=16; MAX_TOK=160; SAMPLE_POS=200   # 每层随机采样这么多位置算两两cos


def main():
    import pandas as pd, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
    stud=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
    teach=AutoModelForCausalLM.from_pretrained(TEACHER,torch_dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
    core=stud.model; nlayer=len(core.layers); H=stud.config.hidden_size
    WU=core.embed_tokens.weight.float().cpu()   # [V,H] tied

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

    # 收集每层 hidden
    layer_h=[[] for _ in range(nlayer)]; handles=[]
    def mk(L):
        def hook(m,i,o):
            h=o[0] if isinstance(o,tuple) else o
            layer_h[L].append(h.detach()[0].float().cpu())
        return hook
    for L,layer in enumerate(core.layers): handles.append(layer.register_forward_hook(mk(L)))
    # 同时拿 student/teacher 的最终 logits(每位置) 算分布差
    stud_logits=[]; teach_logits=[]
    with torch.no_grad():
        for p in prompts:
            ids=tok(p,return_tensors="pt",truncation=True,max_length=MAX_TOK).to("cuda:0")
            so=stud(**ids).logits[0].float().cpu()      # [T,V]
            stud_logits.append(so)
            to=teach(**ids).logits[0].float().cpu()
            teach_logits.append(to)
    for h in handles: h.remove()

    # 对每层: g_t = W_U^T (p_teacher,t - p_student,t)  (高层理想梯度近似, 与层无关的分布差部分)
    # 但我们要看"若在第L层注入", 需要该层的分布差; 简化: 用最终层分布差(reverse-KL目标本身),
    # 这度量的是"输出侧需求的位置冲突", 各层通用。再叠加各层 hidden 的可分性。
    # 计算全局(最终输出)位置梯度 g_t, 及其两两 cos。
    def pos_consistency(vecs):
        # vecs: [M,H] tensor, 归一化后算平均两两 cos
        v=vecs/ (vecs.norm(dim=-1,keepdim=True)+1e-9)
        M=v.shape[0]
        # 平均两两cos = (‖sum‖^2/M^2*... ) 用均值向量法: mean cos ≈ ‖mean(v)‖^2 (对单位向量近似下界)
        # 精确: (Σ_i≠j v_i·v_j)/(M(M-1)) = (‖Σv‖^2 - M)/(M(M-1))
        s=v.sum(0)
        return float((s@s - M)/(M*(M-1)+1e-9))

    # 采样位置构造 g_t (基于最终输出分布差)
    all_g=[]
    for si in range(len(prompts)):
        ps=torch.softmax(stud_logits[si],-1); pt=torch.softmax(teach_logits[si],-1)
        diff=(pt-ps)                      # [T,V]
        g=diff@WU                          # [T,H]  W_U^T(pt-ps)
        all_g.append(g)
    G=torch.cat(all_g,0)                   # [Ntok,H]
    # 随机采样位置
    idx=torch.randperm(G.shape[0])[:SAMPLE_POS]
    g_cons=pos_consistency(G[idx])
    print(f"[output-side] per-position ideal-grad direction consistency (mean pairwise cos) = {g_cons:.4f}")
    print("  (低=各位置想要的输出改动方向互相冲突 -> 单个全局向量难满足)")

    # 各层: 若"直接在该层注入去改输出", 位置冲突度 = 用该层 hidden 的 argmax-onehot 目标增量
    # 更贴合: 每层 hidden 经 logit-lens 得该层"当前预测", 目标是推向 next actual token
    rows=[]
    for L in range(nlayer):
        Hs=torch.cat(layer_h[L],0)         # [Ntok,H]
        # 该层 logit-lens 分布
        rms=Hs/ torch.sqrt((Hs*Hs).mean(-1,keepdim=True)+1e-6)
        logit=rms@WU.T                     # [Ntok,V]
        p=torch.softmax(logit,-1)
        # 目标: 提高每位置 top1 的概率 -> 梯度方向 W_U^T(onehot_top1 - p)
        top1=logit.argmax(-1)
        oneh=torch.zeros_like(p); oneh[torch.arange(p.shape[0]),top1]=1.0
        gL=(oneh-p)@WU                     # [Ntok,H] 该层"直接改输出"的位置梯度
        idx=torch.randperm(gL.shape[0])[:SAMPLE_POS]
        cons=pos_consistency(gL[idx])
        # 对照: 该层 hidden 本身的位置方向一致性(hidden 各向异性baseline)
        hcons=pos_consistency(Hs[idx])
        rows.append({"layer":L,"grad_pos_consistency":round(cons,4),"hidden_pos_consistency":round(hcons,4)})
    with open(os.path.join(OUT,"grad_conflict_by_layer.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nlayer  grad_pos_consistency  hidden_pos_consistency")
    for r in rows: print(f"  {r['layer']:>2}  {r['grad_pos_consistency']:>18.4f}  {r['hidden_pos_consistency']:>18.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ls=[r["layer"] for r in rows]
    fig,ax=plt.subplots(figsize=(8,4.6))
    ax.plot(Ls,[r["grad_pos_consistency"] for r in rows],"o-",color="crimson",label="direct-output grad (per-pos)")
    ax.plot(Ls,[r["hidden_pos_consistency"] for r in rows],"o-",color="gray",alpha=0.6,label="hidden direction (baseline)")
    ax.axhline(0,ls=":",color="k")
    ax.set_xlabel("layer"); ax.set_ylabel("mean pairwise cos across positions")
    ax.set_title("Per-position gradient consistency vs layer\nlow value = positions want conflicting updates → global vector fights itself")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIG,"fig12_grad_conflict.png"),dpi=130); print("saved fig12")

if __name__=="__main__": main()
