#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-PROMPT token hidden states (layers 5..15) and compute the paper-style
matrix-based entropy / effective rank of the token x token Gram matrix, per layer.

Reproduces the "compression valley" of 'Layer by Layer: Uncovering Hidden Representations
in Language Models': for each prompt, at each layer, take the token embeddings H (T x d),
form the (normalized) Gram / covariance, and compute:
  - matrix-based entropy  S = -sum_i (lam_i/sum lam) log(lam_i/sum lam)
  - effective rank        erank = exp(S)   (a.k.a. participation-style, alpha=1 Renyi)
Average over prompts -> one curve over layers. Intermediate layers are expected to dip
(compression valley), unlike the feature-covariance rank which is monotone.

We compute both WITH and WITHOUT the top massive-activation direction removed, because the
dominant "rogue" component otherwise flattens the curve.

Output: analysis/vector_study/cache/layer_entropy_<domain>.pt   (keys: erank, erank_notop, S, S_notop)
Usage:  python extract_layer_entropy.py --domain physics --max_prompts 60
"""
import os, argparse
import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_ROOT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/sciknoweval"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
LAYERS = list(range(5, 16))


def build_prompt(tok, prompt_field):
    msgs = [{"role": m["role"], "content": m["content"]} for m in prompt_field]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def matrix_entropy(H, drop_top=False):
    """H: (T, d) token embeddings for one prompt at one layer.
    Compute eigenvalues of the (T x T) Gram of centered, L2-normalized rows, then
    matrix-based (von Neumann) entropy and effective rank = exp(entropy)."""
    H = H.astype(np.float64)
    H = H - H.mean(0, keepdims=True)
    n = np.linalg.norm(H, axis=1, keepdims=True)
    H = H / (n + 1e-8)
    # Gram over tokens (T x T); its nonzero spectrum matches the covariance spectrum
    G = H @ H.T
    w = np.linalg.eigvalsh(G)
    w = np.clip(w, 0, None)
    w = np.sort(w)[::-1]
    if drop_top and len(w) > 1:
        w = w[1:]
    s = w.sum()
    if s <= 0:
        return 0.0, 1.0
    p = w / s
    p = p[p > 1e-12]
    S = float(-np.sum(p * np.log(p)))
    return S, float(np.exp(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="physics")
    ap.add_argument("--max_prompts", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATA_ROOT, "sciknoweval_validation.parquet"))
    df = df[df["extra_info"].apply(lambda e: e.get("domain") == args.domain)]
    df = df.head(args.max_prompts).reset_index(drop=True)
    print(f"[{args.domain}] {len(df)} prompts")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    layers = model.model.layers

    # per-prompt: capture each layer's full-sequence hidden state
    cur = {L: None for L in LAYERS}
    handles = []
    def make_hook(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cur[L] = h.detach().float()[0].cpu().numpy()   # (seq, d) batch=1
        return hook
    for L in LAYERS:
        handles.append(layers[L].register_forward_hook(make_hook(L)))

    S = {L: [] for L in LAYERS}; ER = {L: [] for L in LAYERS}
    S2 = {L: [] for L in LAYERS}; ER2 = {L: [] for L in LAYERS}
    with torch.no_grad():
        for i in range(len(df)):
            text = build_prompt(tok, df.iloc[i]["prompt"])
            ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
            model(**ids)          # single forward over the full prompt (no generate/KV-cache)
            for L in LAYERS:
                H = cur[L]
                if H is None or H.shape[0] < 8:
                    continue
                s, er = matrix_entropy(H, drop_top=False)
                s2, er2 = matrix_entropy(H, drop_top=True)
                if np.isfinite(er) and np.isfinite(er2):
                    S[L].append(s); ER[L].append(er); S2[L].append(s2); ER2[L].append(er2)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(df)}")
    for h in handles:
        h.remove()

    out = dict(domain=args.domain, layers=LAYERS,
               S={L: float(np.mean(S[L])) for L in LAYERS},
               erank={L: float(np.mean(ER[L])) for L in LAYERS},
               S_notop={L: float(np.mean(S2[L])) for L in LAYERS},
               erank_notop={L: float(np.mean(ER2[L])) for L in LAYERS})
    torch.save(out, os.path.join(CACHE, f"layer_entropy_{args.domain}.pt"))
    print(f"[ok] saved layer_entropy_{args.domain}.pt")
    print("layer  erank  erank_notop")
    for L in LAYERS:
        print(f"L{L:<4d} {out['erank'][L]:6.2f} {out['erank_notop'][L]:8.2f}")


if __name__ == "__main__":
    main()
