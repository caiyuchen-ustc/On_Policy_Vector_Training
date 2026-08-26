#!/usr/bin/env python3
"""For each domain run, decompose every sequential vector's energy across the hidden-state
PCA spectrum in 4 bands: PC1-5 (principal), PC6-30 (mid), PC31-128 (tail), beyond-PC128
(extreme tail). Shows whether v0 sits deepest in the extreme tail and how it shifts with the
orthogonal sequence — for all four science domains.
"""
import os, glob, re, sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

MP = "/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-1.5B"
VEC_BASE = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors"
DATA_BASE = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/sciknoweval/by_domain"
STEER = list(range(5, 16))
Q = 128

RUNS = {
    "physics":   "seqbasis_physics_layers5-15_lr5e-2",
    "chemistry": "seqbasis_chemistry_layers5-15_lr5e-2",
    "biology":   "seqbasis_biology_layers5-15_lr5e-2",
    "material":  "seqbasis_material_layers5-15_lr5e-2",
}


def switch_done_steps(run_dir, probe=5, maxk=6):
    files = sorted(glob.glob(f"{run_dir}/vectors_step_*.pt"),
                   key=lambda p: int(re.search(r"(\d+)\.pt", p).group(1)))
    steps, norms = [], []
    for f in files:
        st = int(re.search(r"(\d+)\.pt", f).group(1))
        o = torch.load(f, map_location="cpu", weights_only=False)
        steps.append(st); norms.append(float(o[probe].float().norm()))
    sw = [steps[i] for i in range(1, len(norms)) if norms[i] < 0.5 * norms[i - 1]]
    return ([s - 1 for s in sw] + [steps[-1]])[:maxk]


def main():
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True).eval()
    H = model.config.hidden_size; core = model.model
    rand = np.array([5, 25, 98, 0]) / Q
    rand[3] = 1 - rand[:3].sum()

    for run, sub in RUNS.items():
        run_dir = os.path.join(VEC_BASE, sub)
        done = switch_done_steps(run_dir)
        # per-domain prompts
        df = pd.read_parquet(f"{DATA_BASE}/{run}/sciknoweval_train.parquet")
        prompts = []
        for i in range(min(24, len(df))):
            r = df.iloc[i]["prompt"]
            prompts.append(tok.apply_chat_template(list(r), tokenize=False, add_generation_prompt=True)
                           if not isinstance(r, str) else r)
        col = {L: [] for L in STEER}; handles = []
        def mk(L):
            def h(m, i, o): col[L].append((o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, H).cpu())
            return h
        for L in STEER: handles.append(core.layers[L].register_forward_hook(mk(L)))
        with torch.no_grad():
            for p in prompts:
                model(**tok(p, return_tensors="pt", truncation=True, max_length=256).to("cuda:0"))
        for h in handles: h.remove()
        pca = {}
        for L in STEER:
            X = torch.cat(col[L], 0); Xc = X - X.mean(0); q = min(Q, H, X.shape[0] - 1)
            _, S, V = torch.pca_lowrank(Xc, q=q, niter=3); pca[L] = (V, q)
        for L in STEER: col[L].clear()

        print(f"\n=== [{run}] 能量分段 (跨层平均) ===")
        print("向量 | PC1-5 | PC6-30 | PC31-128 | PC128以外")
        for k, st in enumerate(done):
            v = torch.load(f"{run_dir}/vectors_step_{st:07d}.pt", map_location="cpu", weights_only=False)
            seg = np.zeros(4)
            for L in STEER:
                V, q = pca[L]; vn = v[L].float() / v[L].float().norm()
                e = (V.T @ vn) ** 2
                seg[0] += e[:5].sum().item(); seg[1] += e[5:30].sum().item(); seg[2] += e[30:].sum().item()
            seg /= len(STEER); seg[3] = 1 - seg[:3].sum()
            print(f"  v{k}  | {seg[0]:.3f} | {seg[1]:.3f} | {seg[2]:.3f} | {seg[3]:.3f}")
        print(f" rand | {rand[0]:.3f} | {rand[1]:.3f} | {rand[2]:.3f} | {rand[3]:.3f}")


if __name__ == "__main__":
    main()
