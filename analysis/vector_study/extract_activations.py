#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract layer 5..15 OUTPUT hidden states (residual stream) of DeepSeek-1.5B on
SciKnowEval, so we can locate the trained steering vectors relative to the activation
manifold. The steering hook adds its vector to each decoder layer's OUTPUT (output[0]),
so the correct reference frame is the layer output hidden state at response-token
positions.

Saves, per layer L in 5..15:
  mu[L]        : mean activation                       (hidden,)
  pcs[L]       : top-K principal component directions  (K, hidden)
  var[L]       : PCA explained variance per component  (K,)
  X_sample[L]  : a subsample of centered activations    (n_sample, hidden) for plotting

Output: analysis/vector_study/cache/act_stats_<domain>.pt
Usage:  python extract_activations.py --domain physics --max_prompts 200
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
TOPK = 64  # principal components to keep


def build_prompt(tok, prompt_field):
    msgs = [{"role": m["role"], "content": m["content"]} for m in prompt_field]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="physics")
    ap.add_argument("--max_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n_sample", type=int, default=4000, help="centered acts to store per layer")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    dfile = os.path.join(DATA_ROOT, "by_domain", args.domain, "validation.parquet")
    if not os.path.exists(dfile):
        dfile = os.path.join(DATA_ROOT, "sciknoweval_validation.parquet")
    df = pd.read_parquet(dfile)
    if "extra_info" in df.columns:
        df = df[df["extra_info"].apply(lambda e: e.get("domain") == args.domain)]
    df = df.head(args.max_prompts).reset_index(drop=True)
    print(f"[{args.domain}] {len(df)} prompts from {dfile}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    layers = model.model.layers
    buffers = {L: [] for L in LAYERS}
    handles = []

    def make_hook(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # h: (batch, seq, hidden) — take all positions of the generated part later;
            # here we just collect every position's hidden state (float32 on cpu)
            buffers[L].append(h.detach().float().reshape(-1, h.shape[-1]).cpu())
        return hook

    for L in LAYERS:
        handles.append(layers[L].register_forward_hook(make_hook(L)))

    with torch.no_grad():
        for i in range(len(df)):
            text = build_prompt(tok, df.iloc[i]["prompt"])
            ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
            gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(df)}")

    for h in handles:
        h.remove()

    stats = {"domain": args.domain, "layers": LAYERS, "mu": {}, "pcs": {}, "var": {}, "X_sample": {}}
    rng = np.random.default_rng(0)
    for L in LAYERS:
        X = torch.cat(buffers[L], dim=0).numpy()          # (N, hidden)
        mu = X.mean(0)
        Xc = X - mu
        # PCA via SVD on a subsample for speed if huge
        if Xc.shape[0] > 20000:
            idx = rng.choice(Xc.shape[0], 20000, replace=False)
            Xf = Xc[idx]
        else:
            Xf = Xc
        U, S, Vt = np.linalg.svd(Xf, full_matrices=False)
        var = (S ** 2) / (Xf.shape[0] - 1)
        stats["mu"][L] = torch.tensor(mu)
        stats["pcs"][L] = torch.tensor(Vt[:TOPK])
        stats["var"][L] = torch.tensor(var[:TOPK])
        ns = min(args.n_sample, Xc.shape[0])
        sidx = rng.choice(Xc.shape[0], ns, replace=False)
        stats["X_sample"][L] = torch.tensor(Xc[sidx])
        print(f"  L{L}: N={X.shape[0]} top1 var ratio={var[0]/var.sum():.3f}")

    out = os.path.join(CACHE, f"act_stats_{args.domain}.pt")
    torch.save(stats, out)
    print(f"[ok] saved {out}")


if __name__ == "__main__":
    main()
