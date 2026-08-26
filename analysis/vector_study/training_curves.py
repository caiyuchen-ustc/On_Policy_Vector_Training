#!/usr/bin/env python3
"""
实验1+4: 从训练 log 提取每个单层实验逐 step 的:
  - val-core AIME2024/2025 reward mean@4  (下游任务分数)
  - actor/steer_vec_norm_mean             (向量norm, 看是否失控)
  - actor/steer_grad_norm_mean            (梯度范数)
  - actor/entropy, actor/pg_loss          (训练信号)
按 层范围 分组, 对比高层 vs 中低层的 (a)分数轨迹 (b)norm演化。
"""
import os, re, glob, csv, json
from collections import defaultdict

VERL="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl"
OUT=os.path.join(VERL,"analysis/vector_study/out"); FIG=os.path.join(VERL,"analysis/vector_study/figs")

# 只挑 qwen3-4b single 层实验(有完整轨迹的)
LOGS={
 "L3-4":   "4bopd_3-4single2e-3.log",   # 低
 "L7-8":   "4bopd_7-8single.log",       # 低
 "L15-16": "4bopd_15-16single.log",     # 中
 "L20-21": "4bopd_20-21single.log",     # 中
 "L24-25": "4bopd_24-25single6e-3.log", # 高
 "L32-33": "4bopd_32-33single.log",     # 高
}

def fnum(s):
    m=re.search(r"np\.float64\(([-\d.eE]+)\)|([-\d.eE]+)", s)
    if not m: return None
    return float(m.group(1) or m.group(2))

def parse(path):
    rows=[]
    for line in open(path, errors="ignore"):
        if "steer_vec_norm_mean" not in line and "val-core/AIME2024/reward/mean@4" not in line:
            continue
        st=re.search(r"step:(\d+)", line)
        if not st: continue
        step=int(st.group(1))
        def grab(key):
            m=re.search(re.escape(key)+r":np\.float64\(([-\d.eE]+)\)", line) or re.search(re.escape(key)+r":([-\d.eE]+)", line)
            return float(m.group(1)) if m else None
        rows.append({
            "step":step,
            "aime24":grab("val-core/AIME2024/reward/mean@4"),
            "aime25":grab("val-core/AIME2025/reward/mean@4"),
            "vec_norm":grab("actor/steer_vec_norm_mean"),
            "grad_norm":grab("actor/steer_grad_norm_mean"),
            "entropy":grab("actor/entropy"),
            "pg_loss":grab("actor/pg_loss"),
        })
    # 合并同step(val行和train行可能分开)
    merged={}
    for r in rows:
        s=r["step"]; merged.setdefault(s,{"step":s})
        for k,v in r.items():
            if v is not None: merged[s][k]=v
    return [merged[s] for s in sorted(merged)]

def main():
    all_data={}
    summary=[]
    for tag,fn in LOGS.items():
        p=os.path.join(VERL,fn)
        if not os.path.exists(p):
            print(f"[skip] {fn}"); continue
        d=parse(p)
        if not d: print(f"[empty] {fn}"); continue
        all_data[tag]=d
        # 汇总: 起始/最终分数, norm增长倍数, 最大norm
        a24=[r.get("aime24") for r in d if r.get("aime24") is not None]
        vn=[r.get("vec_norm") for r in d if r.get("vec_norm") is not None]
        summary.append({
            "exp":tag,"n_step":len(d),
            "aime24_start":round(a24[0],4) if a24 else None,
            "aime24_final":round(a24[-1],4) if a24 else None,
            "aime24_best":round(max(a24),4) if a24 else None,
            "aime24_delta":round(a24[-1]-a24[0],4) if len(a24)>1 else None,
            "vecnorm_start":round(vn[0],3) if vn else None,
            "vecnorm_final":round(vn[-1],3) if vn else None,
            "vecnorm_growth":round(vn[-1]/vn[0],2) if vn and vn[0]>1e-6 else None,
            "vecnorm_max":round(max(vn),3) if vn else None,
        })
    with open(os.path.join(OUT,"layer_training_curves.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print("\n=== 每层实验汇总 ===")
    print(f"{'exp':8} {'nstep':>5} {'aime24 start→final(best)':>26} {'Δ':>7} {'vecnorm start→final':>22} {'growth':>7}")
    for s in summary:
        print(f"{s['exp']:8} {s['n_step']:>5} {str(s['aime24_start'])+'→'+str(s['aime24_final'])+'('+str(s['aime24_best'])+')':>26} "
              f"{str(s['aime24_delta']):>7} {str(s['vecnorm_start'])+'→'+str(s['vecnorm_final']):>22} {str(s['vecnorm_growth']):>7}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    order=list(all_data.keys())
    colors={t:cm.coolwarm(i/max(1,len(order)-1)) for i,t in enumerate(order)}
    fig,axes=plt.subplots(1,3,figsize=(16,4.6))
    for tag,d in all_data.items():
        xs=[r["step"] for r in d if r.get("aime24") is not None]
        ys=[r["aime24"] for r in d if r.get("aime24") is not None]
        if xs: axes[0].plot(xs,ys,"o-",ms=3,label=tag,color=colors[tag])
        xs2=[r["step"] for r in d if r.get("vec_norm") is not None]
        ys2=[r["vec_norm"] for r in d if r.get("vec_norm") is not None]
        if xs2: axes[1].plot(xs2,ys2,"o-",ms=3,label=tag,color=colors[tag])
        xs3=[r["step"] for r in d if r.get("grad_norm") is not None]
        ys3=[r["grad_norm"] for r in d if r.get("grad_norm") is not None]
        if xs3: axes[2].plot(xs3,ys3,"o-",ms=3,label=tag,color=colors[tag])
    axes[0].set_title("(1) AIME2024 val reward vs step\n(downstream task score)"); axes[0].set_xlabel("step"); axes[0].set_ylabel("reward mean@4"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_title("(2) steer vector norm vs step\n(high layers: uncontrolled growth?)"); axes[1].set_xlabel("step"); axes[1].set_ylabel("vec norm"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[2].set_title("(3) steer grad norm vs step"); axes[2].set_xlabel("step"); axes[2].set_ylabel("grad norm"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    fig.suptitle("Training dynamics by layer range (blue=low, red=high layers)",fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig11_training_dynamics_by_layer.png"),dpi=130)
    print("saved fig11")

if __name__=="__main__": main()
