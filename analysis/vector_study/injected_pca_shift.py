#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal test: does injecting the trained vector CHANGE the hidden state's
principal-component structure?

For each layer L that has a trained vector v_L:
  1) clean forward  -> collect residual stream at layer L  -> X   [N,H]
  2) steered forward (hook adds v_L at layer L's output)   -> X'  [N,H]
  3) PCA both, compare the top-k principal subspaces:
       - pc1_cos          : |cos(PC1, PC1')|                    top direction stability
       - subspace_overlap : mean |cos| between the two top-k PC subspaces
                            (via principal angles: normalized Frobenius of PC^T PC')
       - lambda_ratio_topk: sum(λ'_topk)/sum(λ_topk)           variance-spectrum change
       - hidden_shift     : ‖mean(X')-mean(X)‖ / ‖mean(X)‖     how much the cloud moved
       - drift_in_topk    : fraction of the injected shift that lies IN the top-k
                            principal subspace (low => shift is off-principal)

Claim to support: injecting v barely rotates the principal axes (pc1_cos≈1,
subspace_overlap≈1) and barely changes the variance spectrum (ratio≈1); what
little the cloud moves is mostly OUTSIDE the principal subspace -> the vector
steers an off-principal direction WITHOUT disturbing the dominant representation.

Env: MODEL_TAG=qwen3-4b (default) | deepseek7b
Rerun:  python injected_pca_shift.py
Output: out/injected_pca_shift_<tag>.csv  (+ printed summary)
"""
import os, glob, re, csv
import torch
import numpy as np

MODELS = {
    "qwen3-4b": dict(
        path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/Qwen3-4B",
        vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/opd/repre_qwen3-4b-opd/trainable_vectors",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
    ),
    "deepseek7b": dict(
        path="/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-7B",
        vec_root="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/repre/ifevalg_opd/trainable_vectors",
        data="/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/ifevalg/ifevalg_train.parquet",
    ),
}
_TAG = os.environ.get("MODEL_TAG", "qwen3-4b")
MODEL_PATH = MODELS[_TAG]["path"]
VEC_ROOT = MODELS[_TAG]["vec_root"]
DATA = MODELS[_TAG]["data"]
OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
os.makedirs(OUT, exist_ok=True)

N_PROMPTS = int(os.environ.get("N_PROMPTS", 32))
MAX_TOK = int(os.environ.get("MAX_TOK", 256))
TOPK = int(os.environ.get("TOPK", 20))     # principal subspace size to compare
QPCA = 128                                  # PCs to estimate


def load_final_vectors():
    best = {}
    for exp_dir in sorted(glob.glob(os.path.join(VEC_ROOT, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        fs = sorted(glob.glob(exp_dir + "/vectors_step_*.pt"),
                    key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
        if not fs:
            continue
        d = torch.load(fs[-1], map_location="cpu")
        for k, v in d.items():
            t = v if torch.is_tensor(v) else v.get("vector")
            if t is None:
                continue
            t = t.float().reshape(-1); L = int(k)
            if L not in best or t.norm() > best[L][1]:
                best[L] = (t, float(t.norm()))
    return {L: t for L, (t, n) in best.items()}


def _pca(X, q):
    mean = X.mean(0)
    Xc = X - mean
    Uu, Ss, Vv = torch.pca_lowrank(Xc, q=q, niter=4)   # Vv:[H,q]
    lam = (Ss ** 2) / (X.shape[0] - 1)
    return mean, Vv, lam


def _subspace_overlap(Va, Vb):
    """mean cos of principal angles between column-spaces of Va,Vb ([H,k])."""
    # orthonormalize (pca_lowrank Vv are ~orthonormal already)
    M = Va.T @ Vb                       # [k,k]
    s = torch.linalg.svdvals(M)          # cos(principal angles)
    return float(s.mean()), float(s.min())


def main():
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vecs = load_final_vectors()
    layers_with_vec = sorted(vecs)
    print(f"[{_TAG}] layers with trained vector:", layers_with_vec)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    core = model.model
    nlayer = len(core.layers)
    H = model.config.hidden_size
    dirs = {L: v.to("cuda:0") for L, v in vecs.items()}  # raw (keep norm) injection

    df = pd.read_parquet(DATA)
    pcol = next((c for c in ["prompt", "question", "problem", "query"] if c in df.columns), df.columns[0])
    prompts = []
    for i in range(min(N_PROMPTS, len(df))):
        row = df.iloc[i][pcol]
        if isinstance(row, (list, np.ndarray)):
            txt = tok.apply_chat_template(list(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            txt = tok.apply_chat_template([{"role": "user", "content": str(row)}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        prompts.append(txt)

    ids_list = [tok(p, return_tensors="pt", truncation=True, max_length=MAX_TOK).to("cuda:0") for p in prompts]

    def collect_pair(L):
        """clean X at layer L+1, and steered X' at L+1 when v is injected at layer L."""
        Lr = min(L + 1, nlayer - 1)
        clean = []
        steered = []
        vdir = dirs[L]
        def grab_clean(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            clean.append(h.detach().float().reshape(-1, h.shape[-1]).cpu())
        def grab_steer(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            steered.append(h.detach().float().reshape(-1, h.shape[-1]).cpu())
        def inject(m, i, o):
            if isinstance(o, tuple):
                return (o[0] + vdir.to(o[0].dtype),) + o[1:]
            return o + vdir.to(o.dtype)
        with torch.no_grad():
            gh = core.layers[Lr].register_forward_hook(grab_clean)
            for ids in ids_list:
                model(**ids)
            gh.remove()
            ih = core.layers[L].register_forward_hook(inject)
            gh = core.layers[Lr].register_forward_hook(grab_steer)
            for ids in ids_list:
                model(**ids)
            gh.remove(); ih.remove()
        return torch.cat(clean, 0), torch.cat(steered, 0)

    rows = []
    for L in layers_with_vec:
        X, Xs = collect_pair(L)
        q = min(QPCA, H, X.shape[0] - 1)
        mean, V, lam = _pca(X, q)
        means, Vs, lams = _pca(Xs, q)
        k = min(TOPK, q)
        Vk, Vsk = V[:, :k], Vs[:, :k]
        mean_cos, min_cos = _subspace_overlap(Vk, Vsk)
        pc1_cos = float(abs(torch.dot(V[:, 0], Vs[:, 0])))
        lam_ratio = float(lams[:k].sum() / lam[:k].sum())
        shift = means - mean
        shift_rel = float(shift.norm() / (mean.norm() + 1e-9))
        # fraction of shift energy inside the clean top-k principal subspace
        proj = Vk.T @ shift
        drift_in_topk = float((proj ** 2).sum() / (shift.dot(shift) + 1e-9))
        vnorm = float(vecs[L].norm())
        rows.append(dict(layer=L, vec_norm=round(vnorm, 3),
                         pc1_cos=round(pc1_cos, 4),
                         subspace_overlap=round(mean_cos, 4),
                         subspace_overlap_min=round(min_cos, 4),
                         lambda_ratio_topk=round(lam_ratio, 4),
                         hidden_shift_rel=round(shift_rel, 4),
                         drift_in_topk=round(drift_in_topk, 4),
                         drift_rand_baseline=round(k / H, 4)))
        print(f"L{L:>2} pc1_cos={pc1_cos:.3f} overlap={mean_cos:.3f} λ'/λ={lam_ratio:.3f} "
              f"shift={shift_rel:.3f} drift_in_top{k}={drift_in_topk:.3f} (rand {k/H:.3f})")

    fn = os.path.join(OUT, f"injected_pca_shift_{_TAG}.csv")
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[ok] wrote {fn}")


if __name__ == "__main__":
    main()
