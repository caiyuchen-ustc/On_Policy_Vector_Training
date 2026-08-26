#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the primary steering vector v0 the compressed form of the teacher's fine-tuning direction?

The steering vector v is ADDED to the residual stream at layer L. If v0 captures what the
teacher's fine-tuning does, then the teacher's per-layer residual-stream SHIFT relative to the
base model, Delta_h_L = mean(h_teacher_L) - mean(h_base_L), should point along v0.

We can't directly compare v (a residual direction, [H]) with Delta_W (a weight matrix), but we
CAN compare v with the *effect* of Delta_W on the residual stream, Delta_h — they live in the
same space. cos(v0, Delta_h) high => v0 is the direction the teacher's weights push the stream.

Method (teacher-forcing on the SAME token ids so positions align):
  for a batch of prompts, run base and teacher, hook each layer L's output h_L, average over
  tokens -> mean_base_L, mean_teacher_L. Delta_h_L = mean_teacher_L - mean_base_L.
  For each steer layer L: cos(v0_L, Delta_h_L), plus cos(v1_L, Delta_h_L), cos(random, Delta_h_L).

Env: none. Edit PATHS below for a different model.
Output: out/teacher_shift_vs_vector.csv (+ printed summary)
"""
import os, glob, re, csv
import torch
import numpy as np

BASE = "/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B"
TEACHER = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_rl/ifevalg_qwen2.5-7b-fullparam_lr1e-6/global_step_825/hf_model"
VEC_DIR = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_offline_distill/trainable_vectors/singlev_layers8-25_lr5e-2"
DATA = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/ifevalg/ifevalg_train.parquet"
OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
os.makedirs(OUT, exist_ok=True)

# vector done-steps for this run (v0 done at 100; v1/v2 = switch-1). Fill from log if needed.
V0_STEP = 100
V1_STEP = 241   # 8-25 run: first switch at 242 -> v1 done at 241


def collect_mean_hidden(model_path, prompt_ids, steer_layers, H, device="cuda:0"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map=device, trust_remote_code=True).eval()
    core = model.model
    acc = {L: torch.zeros(H, dtype=torch.float64) for L in steer_layers}
    cnt = {L: 0 for L in steer_layers}
    handles = []
    def mk(L):
        def hook(m, i, o):
            h = (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, H).cpu().double()
            acc[L] += h.sum(0); cnt[L] += h.shape[0]
        return hook
    for L in steer_layers:
        handles.append(core.layers[L].register_forward_hook(mk(L)))
    with torch.no_grad():
        for ids in prompt_ids:
            model(input_ids=ids.to(device))
    for h in handles:
        h.remove()
    means = {L: (acc[L] / max(cnt[L], 1)) for L in steer_layers}
    del model
    torch.cuda.empty_cache()
    return means


def main():
    import pandas as pd
    from transformers import AutoTokenizer

    v0 = torch.load(f"{VEC_DIR}/vectors_step_{V0_STEP:07d}.pt", map_location="cpu", weights_only=False)
    v1 = torch.load(f"{VEC_DIR}/vectors_step_{V1_STEP:07d}.pt", map_location="cpu", weights_only=False)
    steer_layers = sorted(v0.keys())
    print("steer layers:", steer_layers)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    H = len(v0[steer_layers[0]])

    df = pd.read_parquet(DATA)
    ids_list = []
    for i in range(24):
        row = df.iloc[i]["prompt"]
        txt = (tok.apply_chat_template(list(row), tokenize=False, add_generation_prompt=True)
               if not isinstance(row, str) else row)
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=256).input_ids
        ids_list.append(ids)

    print("collecting base hidden means...")
    base_means = collect_mean_hidden(BASE, ids_list, steer_layers, H)
    print("collecting teacher hidden means...")
    teach_means = collect_mean_hidden(TEACHER, ids_list, steer_layers, H)

    def cos(a, b):
        a = a.double(); b = b.double()
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

    torch.manual_seed(0)
    rows = []
    print("\nlayer | cos(v0, Δh) | cos(v1, Δh) | cos(rand, Δh) | ||Δh|| | ||v0||")
    for L in steer_layers:
        dh = (teach_means[L] - base_means[L])
        r = torch.randn(H, dtype=torch.float64)
        c0 = cos(v0[L].double(), dh)
        c1 = cos(v1[L].double(), dh) if L in v1 else float("nan")
        cr = cos(r, dh)
        rows.append({"layer": L, "cos_v0_dh": round(c0, 4), "cos_v1_dh": round(c1, 4),
                     "cos_rand_dh": round(cr, 4), "dh_norm": round(float(dh.norm()), 4),
                     "v0_norm": round(float(v0[L].double().norm()), 4)})
        print(f"  L{L:2d} | {c0:+.4f} | {c1:+.4f} | {cr:+.4f} | {float(dh.norm()):8.3f} | {float(v0[L].double().norm()):7.3f}")

    import statistics
    m0 = statistics.mean(abs(r["cos_v0_dh"]) for r in rows)
    m1 = statistics.mean(abs(r["cos_v1_dh"]) for r in rows if r["cos_v1_dh"] == r["cos_v1_dh"])
    mr = statistics.mean(abs(r["cos_rand_dh"]) for r in rows)
    print(f"\n平均 |cos|:  v0·Δh = {m0:.4f}   v1·Δh = {m1:.4f}   random·Δh = {mr:.4f}")
    print(f"(高维随机基线 ~ 1/sqrt({H}) = {1/np.sqrt(H):.4f})")

    with open(f"{OUT}/teacher_shift_vs_vector.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT}/teacher_shift_vs_vector.csv")


if __name__ == "__main__":
    main()
