#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project each domain's full orthogonal basis v0..v_k onto that domain's activation PCs,
across all four SciKnowEval domains. Confirms the physics finding is universal:
  - vectors are ~orthogonal to the activation principal axis PC1 (which holds ~99% variance),
  - they live in the low-variance side directions,
  - injection magnitude is gentle (z-score w.r.t. per-direction activation std << 1).
Prints a compact per-domain table; returns aggregate stats for figure making.
"""
import torch, numpy as np, glob, os, re

ROOT = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/deepseek1p5b_offline_distill/trainable_vectors"
DIRS = {"physics": "seqbasis_physics_layers5-15_lr5e-2",
        "chemistry": "seqbasis_chemistry_layers5-15_lr5e-2",
        "biology": "seqbasis_biology_layers5-15_lr5e-2",
        "material": "seqbasis_material_layers5-15_lr5e-2"}
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
LY = list(range(5, 16))


def extract_basis(base):
    steps = sorted(int(re.findall(r"(\d+)", os.path.basename(f))[0])
                   for f in glob.glob(f"{base}/vectors_step_*.pt"))
    def load(s):
        o = torch.load(f"{base}/vectors_step_{s:07d}.pt", map_location="cpu", weights_only=False)
        return {L: o[L].float().numpy() for L in LY}
    def flat(d): return np.concatenate([d[L] for L in LY])
    prev = flat(load(steps[0])); sw = []
    for s in steps[1:]:
        v = flat(load(s)); c = float(prev @ v / (np.linalg.norm(prev) * np.linalg.norm(v) + 1e-9))
        if c < 0.5: sw.append(s)
        prev = v
    ends = [s - 1 for s in sw] + [steps[-1]]
    return [load(e) for e in ends]


def main():
    print(f"{'domain':10s} {'K':>3s} {'PC1var%':>8s} {'|cos(v,PC1)|':>13s} {'align_top8':>11s} {'max_z':>7s}")
    agg = {}
    for dom, sub in DIRS.items():
        st = torch.load(os.path.join(CACHE, f"act_stats_{dom}.pt"), weights_only=False)
        V = extract_basis(f"{ROOT}/{sub}")
        # aggregate over layers 5..15
        cos_pc1_all, align8_all, maxz_all, pc1var = [], [], [], []
        per_vec_cos_pc1 = np.zeros(len(V)); cnt = 0
        for L in LY:
            mu = st["mu"][L].numpy(); pcs = st["pcs"][L].numpy(); var = st["var"][L].numpy()
            sd = np.sqrt(var); pc1 = pcs[0]
            pc1var.append(var[0] / var.sum())
            for k, vd in enumerate(V):
                vk = vd[L]; n = np.linalg.norm(vk); u = vk / n
                cos_pc1_all.append(abs(u @ pc1))
                per_vec_cos_pc1[k] += abs(u @ pc1)
                align8_all.append(float(np.sum((pcs[:8] @ u) ** 2)))
                z = np.abs(pcs @ vk) / sd
                maxz_all.append(z.max())
            cnt += 1
        per_vec_cos_pc1 /= cnt
        print(f"{dom:10s} {len(V):3d} {np.mean(pc1var)*100:7.2f}% "
              f"{np.mean(cos_pc1_all):13.3f} {np.mean(align8_all):11.4f} {np.mean(maxz_all):7.2f}")
        agg[dom] = dict(K=len(V), pc1var=np.mean(pc1var),
                        cos_pc1=np.mean(cos_pc1_all), align8=np.mean(align8_all),
                        maxz=np.mean(maxz_all), per_vec_cos_pc1=per_vec_cos_pc1)
    torch.save(agg, os.path.join(CACHE, "projection_agg.pt"))
    print(f"\n[ok] saved {os.path.join(CACHE, 'projection_agg.pt')}")
    # sanity: random-vector baseline for cos(v, PC1)
    d = 1536
    rand_cos = np.mean([abs(np.random.default_rng(i).standard_normal(d) @
                            (lambda x: x/np.linalg.norm(x))(np.random.default_rng(i+99).standard_normal(d)))
                        / np.linalg.norm(np.random.default_rng(i).standard_normal(d)) for i in range(200)])
    print(f"random |cos| baseline (per-layer d={d}) ~ {1/np.sqrt(d):.4f}")


if __name__ == "__main__":
    main()
