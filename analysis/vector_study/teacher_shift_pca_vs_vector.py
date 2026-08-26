#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0 vs the PRINCIPAL direction of the teacher's per-token residual-stream shift.

Earlier we compared v0 to the MEAN shift mean(h_teacher)-mean(h_base): cos was only ~0.05
(the mean cancels a lot). Here we instead take the per-token shift delta_t = h_teacher_t -
h_base_t, PCA the set {delta_t}, and compare v0 to its top principal directions PC1..PC5.
Rationale: if the teacher steers every token along a consistent low-dim direction, that shows
up as the top PC of the per-token shifts (not necessarily their mean). cos(v0, PC1) high =>
v0 IS the dominant direction along which the teacher's weights move the residual stream.

Teacher-forcing on identical token ids so base/teacher positions align.
Output: out/teacher_shift_pca_vs_vector.csv
"""
import os, csv
import torch
import numpy as np

BASE = "/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B"
TEACHER = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_rl/ifevalg_qwen2.5-7b-fullparam_lr1e-6/global_step_825/hf_model"
VEC_DIR = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/ifevalg_offline_distill/trainable_vectors/singlev_layers8-25_lr5e-2"
DATA = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/ifevalg/ifevalg_train.parquet"
OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
os.makedirs(OUT, exist_ok=True)
V0_STEP, V1_STEP = 100, 241


def collect_perlayer_hidden(model_path, ids_list, steer_layers, H, device="cuda:0"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map=device, trust_remote_code=True).eval()
    core = model.model
    per = {L: [] for L in steer_layers}
    handles = []
    def mk(L):
        def hook(m, i, o):
            h = (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, H).cpu()
            per[L].append(h)
        return hook
    for L in steer_layers:
        handles.append(core.layers[L].register_forward_hook(mk(L)))
    with torch.no_grad():
        for ids in ids_list:
            per_mark = {L: len(per[L]) for L in steer_layers}
            model(input_ids=ids.to(device))
    for h in handles:
        h.remove()
    out = {L: torch.cat(per[L], 0) for L in steer_layers}   # [Ntok, H] aligned across prompts
    del model; torch.cuda.empty_cache()
    return out


def main():
    import pandas as pd
    from transformers import AutoTokenizer
    v0 = torch.load(f"{VEC_DIR}/vectors_step_{V0_STEP:07d}.pt", map_location="cpu", weights_only=False)
    v1 = torch.load(f"{VEC_DIR}/vectors_step_{V1_STEP:07d}.pt", map_location="cpu", weights_only=False)
    steer_layers = sorted(v0.keys())
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    H = len(v0[steer_layers[0]])

    df = pd.read_parquet(DATA)
    ids_list = []
    for i in range(24):
        row = df.iloc[i]["prompt"]
        txt = (tok.apply_chat_template(list(row), tokenize=False, add_generation_prompt=True)
               if not isinstance(row, str) else row)
        ids_list.append(tok(txt, return_tensors="pt", truncation=True, max_length=256).input_ids)

    print("collecting base per-token hidden...")
    Hb = collect_perlayer_hidden(BASE, ids_list, steer_layers, H)
    print("collecting teacher per-token hidden...")
    Ht = collect_perlayer_hidden(TEACHER, ids_list, steer_layers, H)

    def cosd(a, b):
        a = a.double(); b = b.double(); return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

    torch.manual_seed(0)
    rows = []
    print("\nlayer | cos(v0,PC1) cos(v0,PC1-5best) | cos(v0,meanΔ) | PC1 var-frac | cos(rand,PC1)")
    for L in steer_layers:
        D = (Ht[L] - Hb[L]).double()          # [Ntok, H] per-token shift
        meanD = D.mean(0)
        Dc = D - meanD                          # center so PCA captures the *spread* direction
        q = min(16, H, Dc.shape[0] - 1)
        U, S, V = torch.pca_lowrank(Dc, q=q, niter=4)   # V:[H,q]
        varfrac = float((S[0] ** 2) / (S ** 2).sum())
        v0L = v0[L].double()
        c_pc1 = abs(cosd(v0L, V[:, 0]))
        c_best = max(abs(cosd(v0L, V[:, k])) for k in range(q))
        c_mean = cosd(v0L, meanD)
        r = torch.randn(H, dtype=torch.float64)
        c_rand = abs(cosd(r, V[:, 0]))
        rows.append({"layer": L, "cos_v0_pc1": round(c_pc1, 4), "cos_v0_pc1to5_best": round(c_best, 4),
                     "cos_v0_meanshift": round(c_mean, 4), "pc1_var_frac": round(varfrac, 4),
                     "cos_rand_pc1": round(c_rand, 4)})
        print(f"  L{L:2d} | {c_pc1:.4f}  {c_best:.4f} | {c_mean:+.4f} | {varfrac:.3f} | {c_rand:.4f}")

    import statistics
    print(f"\n平均: cos(v0,PC1)={statistics.mean(r['cos_v0_pc1'] for r in rows):.4f}  "
          f"cos(v0,best-of-PC1-5..)={statistics.mean(r['cos_v0_pc1to5_best'] for r in rows):.4f}  "
          f"cos(rand,PC1)={statistics.mean(r['cos_rand_pc1'] for r in rows):.4f}")
    print(f"(随机基线 ~ 1/sqrt({H}) = {1/np.sqrt(H):.4f})")
    with open(f"{OUT}/teacher_shift_pca_vs_vector.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {OUT}/teacher_shift_pca_vs_vector.csv")


if __name__ == "__main__":
    main()
