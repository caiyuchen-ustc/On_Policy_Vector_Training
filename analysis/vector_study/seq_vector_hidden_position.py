#!/usr/bin/env python3
"""How do the SEQUENTIAL steer vectors (v0, v1, v2, ...) sit in the hidden-state PCA spectrum,
and does that position drift as the orthogonal sequence proceeds?

For the 1.5B seqbasis runs (layers 5-15), for each run:
  1) run the FROZEN base model on sciknoweval prompts, collect per-layer residual-stream
     hidden states X_L [N, H], PCA them (SVD) -> principal components PC_k, variances lam_k.
  2) extract each sequential vector v_k^{(L)} at its "done" step (v0=step100, v_k=switch_step-1).
  3) for each (vector index k, layer L) measure where v sits in the spectrum:
       - pc_centroid_rank : sum(proj_k^2 * rank_k) / sum(proj_k^2)   (low=principal, high=tail)
       - E_top{1,5,20}    : energy fraction of v in the top-k PCs (>k/H => aligned w/ main modes)
       - cos_pc1          : |cos(v, PC1)|
  4) report how centroid_rank / E_top20 change across k=0,1,2,... (does later v move to tail?).

Usage:
  python3 seq_vector_hidden_position.py --run mixed --n_prompts 32
  python3 seq_vector_hidden_position.py --run physics
  (runs: mixed, physics, chemistry, biology, material)
"""
import os, glob, re, csv, argparse
import torch
import numpy as np

MODEL_PATH = "/apdcephfs_zwfy3/share_302867165/xxucaxu/models/raw/DeepSeek-R1-Distill-Qwen-1.5B"
VEC_BASE = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors"
DATA = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/data/sciknoweval/sciknoweval_train.parquet"
OUT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/analysis/vector_study/out"
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "mixed":     "singlev_layers5-15_lr5e-2",
    "physics":   "seqbasis_physics_layers5-15_lr5e-2",
    "chemistry": "seqbasis_chemistry_layers5-15_lr5e-2",
    "biology":   "seqbasis_biology_layers5-15_lr5e-2",
    "material":  "seqbasis_material_layers5-15_lr5e-2",
}


def find_switch_steps(run_dir, probe_layer):
    """Return the list of 'vector done' steps: v0 done at 100, v_k done at switch_step-1."""
    files = sorted(glob.glob(f"{run_dir}/vectors_step_*.pt"),
                   key=lambda p: int(re.search(r"(\d+)\.pt", p).group(1)))
    steps, norms = [], []
    for f in files:
        st = int(re.search(r"(\d+)\.pt", f).group(1))
        o = torch.load(f, map_location="cpu", weights_only=False)
        if probe_layer in o:
            steps.append(st); norms.append(float(o[probe_layer].float().norm()))
    switches = [steps[i] for i in range(1, len(norms)) if norms[i] < 0.5 * norms[i - 1]]
    # vector k finishes the step BEFORE the (k+1)-th switch; last vector = last saved step
    done = [s - 1 for s in switches] + [steps[-1]]
    return done, steps


def load_vec_at_step(run_dir, step):
    # find the exact file for `step` (or nearest <= step)
    f = f"{run_dir}/vectors_step_{step:07d}.pt"
    if not os.path.exists(f):
        cand = sorted(glob.glob(f"{run_dir}/vectors_step_*.pt"),
                      key=lambda p: int(re.search(r"(\d+)\.pt", p).group(1)))
        cand = [c for c in cand if int(re.search(r"(\d+)\.pt", c).group(1)) <= step]
        if not cand:
            return None
        f = cand[-1]
    return torch.load(f, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mixed", choices=list(RUNS))
    ap.add_argument("--n_prompts", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_vectors", type=int, default=6, help="how many sequential vectors to analyze")
    args = ap.parse_args()

    run_dir = os.path.join(VEC_BASE, RUNS[args.run])
    # probe layer = first steer layer (5)
    sample = torch.load(sorted(glob.glob(f"{run_dir}/vectors_step_*.pt"),
                        key=lambda p: int(re.search(r"(\d+)\.pt", p).group(1)))[0],
                        map_location="cpu", weights_only=False)
    steer_layers = sorted(sample.keys())
    done_steps, _ = find_switch_steps(run_dir, steer_layers[0])
    done_steps = done_steps[:args.max_vectors]
    print(f"[{args.run}] steer layers {steer_layers[0]}-{steer_layers[-1]}, "
          f"vector done-steps: {done_steps}")

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size

    # prompts from sciknoweval train (chat -> text)
    df = pd.read_parquet(DATA)
    prompts = []
    for i in range(min(args.n_prompts, len(df))):
        row = df.iloc[i]["prompt"]
        if isinstance(row, (list, np.ndarray)):
            txt = tok.apply_chat_template(list(row), tokenize=False, add_generation_prompt=True)
        else:
            txt = str(row)
        prompts.append(txt)

    core = model.model
    collected = {L: [] for L in steer_layers}
    handles = []
    def mk(L):
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            collected[L].append(h.detach().float().reshape(-1, h.shape[-1]).cpu())
        return hook
    for L in steer_layers:
        handles.append(core.layers[L].register_forward_hook(mk(L)))
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=args.max_tokens).to("cuda:0")
            model(**ids)
    for h in handles:
        h.remove()

    # PCA per steer layer
    pca = {}
    for L in steer_layers:
        X = torch.cat(collected[L], 0)
        N = X.shape[0]
        Xc = X - X.mean(0)
        q = min(128, H, N - 1)
        _, Ss, Vv = torch.pca_lowrank(Xc, q=q, niter=3)
        lam = (Ss ** 2) / (N - 1)
        pca[L] = (Vv, lam, q, N)

    # for each vector index, each layer: position metrics
    rows = []
    for k, step in enumerate(done_steps):
        vecs = load_vec_at_step(run_dir, step)
        if vecs is None:
            continue
        for L in steer_layers:
            if L not in vecs:
                continue
            Vv, lam, q, N = pca[L]
            v = vecs[L].float().reshape(-1)
            vn = v / (v.norm() + 1e-9)
            proj = Vv.T @ vn                      # [q]
            e = proj ** 2
            ranks = torch.arange(1, q + 1, dtype=torch.float32)
            centroid = float((e * ranks).sum() / (e.sum() + 1e-12))
            rec = {"run": args.run, "vec_idx": k, "step": step, "layer": L,
                   "vec_norm": round(float(v.norm()), 3),
                   "pc_centroid_rank": round(centroid, 2),
                   "pc_centroid_rand": round((q + 1) / 2, 1),
                   "cos_pc1": round(float(abs(proj[0])), 4)}
            cum = torch.cumsum(e, 0)
            for kk in [1, 5, 20]:
                if kk <= q:
                    rec[f"E_top{kk}"] = round(float(cum[kk - 1]), 4)
                    rec[f"E_top{kk}_over_rand"] = round(float(cum[kk - 1]) / (kk / H), 2)
            rows.append(rec)

    out_csv = os.path.join(OUT, f"seq_vec_hidden_pos_{args.run}.csv")
    keys = ["run", "vec_idx", "step", "layer", "vec_norm", "pc_centroid_rank", "pc_centroid_rand",
            "cos_pc1", "E_top1", "E_top1_over_rand", "E_top5", "E_top5_over_rand",
            "E_top20", "E_top20_over_rand"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # summary: mean over layers, per vector index
    print(f"\n=== [{args.run}] 各向量在 hidden PCA 谱里的位置 (跨层平均) ===")
    print(f"H={H}, 随机向量 centroid_rank≈{(pca[steer_layers[0]][2]+1)/2:.0f}, E_top20/rand≈1.0")
    print("vec | 跨层平均 centroid_rank | E_top1/rand | E_top5/rand | E_top20/rand | cos_pc1")
    for k in sorted(set(r["vec_idx"] for r in rows)):
        sub = [r for r in rows if r["vec_idx"] == k]
        cr = np.mean([r["pc_centroid_rank"] for r in sub])
        e1 = np.mean([r.get("E_top1_over_rand", 0) for r in sub])
        e5 = np.mean([r.get("E_top5_over_rand", 0) for r in sub])
        e20 = np.mean([r.get("E_top20_over_rand", 0) for r in sub])
        c1 = np.mean([r["cos_pc1"] for r in sub])
        print(f" v{k}  |  {cr:6.1f}  |  {e1:5.2f}  |  {e5:5.2f}  |  {e20:5.2f}  |  {c1:.3f}")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
