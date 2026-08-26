#!/usr/bin/env python3
"""
补充2: teacher 行为对照。
  用 vLLM 让 teacher(Qwen3-4B-Non-Thinking-RL-Math-Step500) 和 base student(Qwen3-4B)
  各自对同一批数学 prompt 生成, 统计两类过渡词的出现频率:
    - reflection (高层向量的目标): Wait/But/Hmm/Alternatively/Perhaps/但/然而...
    - transition (中低层 push-through 的目标): Okay/Alright/So/</think>...
  看向量在模仿 teacher 相对 base 多出来的哪部分行为。
"""
import os, re, json, collections
import pandas as pd, numpy as np

MODELS={
    "qwen3-4b": dict(
        teacher="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B-Non-Thinking-RL-Math-Step500",
        base="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
    ),
    "deepseek7b": dict(
        teacher="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_rl/ifevalg_qwen2.5-7b-fullparam_lr1e-6/global_step_825/hf_model",
        base="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/ifevalg/ifevalg_train.parquet",
    ),
}
_TAG=os.environ.get("MODEL_TAG","qwen3-4b")
TEACHER=MODELS[_TAG]["teacher"]; BASE=MODELS[_TAG]["base"]; DATA=MODELS[_TAG]["data"]
OUT="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
FIG="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/figs"
N=64; MAXGEN=1024

REFLECT=[r"\bWait\b",r"\bBut\b",r"\bHmm+\b",r"\bAlternatively\b",r"\bPerhaps\b",r"\bMaybe\b",
         r"\bHowever\b","但","然而","或许","嗯","不过"]
TRANS=[r"\bOkay\b",r"\bOK\b",r"\bAlright\b",r"\bSo\b",r"</think>",r"\bLet me\b",r"\bFirst\b"]

def count(texts, pats):
    tot_tok=sum(len(t.split()) for t in texts)+1
    c=collections.Counter()
    per=[]
    for t in texts:
        n=sum(len(re.findall(p,t)) for p in pats); per.append(n)
    # 每千词频率 + 每条命中率
    total=sum(per)
    return total/ (tot_tok/1000.0), np.mean([1 if x>0 else 0 for x in per]), total

def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    df=pd.read_parquet(DATA)
    pcol=next((c for c in ["prompt","question","problem","query"] if c in df.columns),df.columns[0])
    tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
    prompts=[]
    for i in range(N):
        row=df.iloc[i][pcol]
        if isinstance(row,(list,np.ndarray)):
            txt=tok.apply_chat_template(list(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        else:
            txt=tok.apply_chat_template([{"role":"user","content":str(row)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompts.append(txt)
    sp=SamplingParams(temperature=0.8,top_p=0.95,max_tokens=MAXGEN,n=1)

    results={}
    for tag,path in [("teacher",TEACHER),("base",BASE)]:
        llm=LLM(model=path,dtype="bfloat16",gpu_memory_utilization=0.85,
                tensor_parallel_size=1,max_model_len=4096,enforce_eager=True,trust_remote_code=True)
        outs=llm.generate(prompts,sp)
        gens=[o.outputs[0].text for o in outs]
        json.dump(gens,open(os.path.join(OUT,f"gen_{tag}_{_TAG}.json"),"w"))
        r_kilo,r_hit,r_tot=count(gens,REFLECT)
        t_kilo,t_hit,t_tot=count(gens,TRANS)
        results[tag]=dict(reflect_per1k=round(r_kilo,3),reflect_hitrate=round(r_hit,3),reflect_total=r_tot,
                          trans_per1k=round(t_kilo,3),trans_hitrate=round(t_hit,3),trans_total=t_tot,
                          n_gen=len(gens),avg_words=round(np.mean([len(g.split()) for g in gens]),1))
        print(tag,results[tag])
        del llm
        import gc,torch; gc.collect(); torch.cuda.empty_cache()

    json.dump(results,open(os.path.join(OUT,f"teacher_behavior_{_TAG}.json"),"w"),indent=2)
    print(json.dumps(results,indent=2,ensure_ascii=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels=["reflection\n(Wait/But/Hmm)","transition\n(Okay/Alright/So)"]
    x=np.arange(2); w=0.35
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.bar(x-w/2,[results["teacher"]["reflect_per1k"],results["teacher"]["trans_per1k"]],w,label="teacher (RL-Math)",color="crimson")
    ax.bar(x+w/2,[results["base"]["reflect_per1k"],results["base"]["trans_per1k"]],w,label="base student",color="gray")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("occurrences per 1k words")
    ax.set_title("Teacher vs base: which transition/reflection words RL added\n(高层向量目标=reflection, 中低层push-through目标=transition)")
    ax.legend(); ax.grid(alpha=0.3,axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(FIG,f"fig9_teacher_behavior_{_TAG}.png"),dpi=130); print("saved fig9")

if __name__=="__main__": main()
