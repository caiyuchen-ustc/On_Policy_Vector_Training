# Section: What Does RL Learn, and Where Does It Live?
## A Layer-wise Steering Study

> ICLR 正文 section 草稿（B 结构：claim-first + puzzle-solving）。
> 承接 Finding 1（RLVR 的增益可被 on-policy 的 OPD/RFT 用一组 steering 向量大部分捕获，
> SFT / off-policy 蒸馏则不行）。
> 结构：X.1 puzzle（全层三方对比）→ X.2 claim + 正面证据（中低层 = 改推理模式）
> → X.3 边界（高层失效，压缩 1 段 + 指向附录）→ X.4 刻画 RL 能力所处的空间。
> 深度机制实验全部搬到附录 C，正文点到为止。

---

## X.1 Setup and a Puzzle: Steering Is Layer-Selective, Full-param/LoRA Is Not

Finding 1 established that RLVR's gain over the base model can be **largely recovered by a small set
of additive steering vectors** trained with an on-policy distillation objective (OPD/RFT), while
SFT / off-policy distillation cannot. This implies the gain has a **low-dimensional, additive**
structure. We now ask **where in the network this structure lives**, by restricting three
fine-tuning methods to *low*, *mid*, and *high* layers and asking which can still recover the gain.

**表 1（正文主表，需填入 full/LoRA 实测分数）**

| Method | What it changes | Low layers | Mid layers | High layers |
|---|---|:--:|:--:|:--:|
| Full-parameter | that layer's weights | ✓ | ✓ | ✓ |
| LoRA | low-rank weight update | ✓ | ✓ | ✓ |
| **Steering vector** | additive direction in residual | ✓ | ✓ | **✗** |

The puzzle is stark: methods that modify a layer's **computation** (full-param, LoRA) work
*everywhere*; the additive **steering vector** works only in the lower two-thirds of the network and
**fails in the high layers**. The failure is not marginal — it is a training pathology (Fig. 1):
mid/low-layer steering raises downstream accuracy (AIME reward **+0.33**) with a stably converging
vector norm, whereas high-layer steering either **blows up in norm** (L24–25 → 16.9, ~6× other
layers) or **stalls with vanishing gradients** (L32–33, Δ +0.03, norm stuck at 0.5).

Crucially, high layers are **not devoid of RL signal** — full-param and LoRA recover the gain there
just fine. So the steering failure is specific to the **additive-direction form**, and its
layer-selectivity is exactly what lets us use steering as a **probe**: *the layers where steering
works are the layers where the RL gain exists as an additive direction.* This section turns that
probe into an answer about what RL learns and where.

---

## X.2 Claim: In Low/Mid Layers, the Steering Vector Encodes a **Reasoning Computation Pattern**

We first establish the positive claim, then (X.3) use it to explain the high-layer failure.

**Claim.** In low/mid layers, the trained steering direction does **not** target the output
vocabulary; it lies in a subspace **orthogonal to the token-output directions** and instead
**modulates an internal reasoning pattern** that is realized by the downstream layers.

Three pieces of evidence support this (full analysis in App. C.1; two models — Qwen3-4B math OPD and
DeepSeek-R1-Distill-Qwen-7B instruction-following OPD — give the same picture, App. C.5):

1. **The direction is vocabulary-agnostic.** Projecting each layer's vector to the vocabulary via
   the logit lens (`RMSNorm(v)·W_Uᵀ`) yields, for all low/mid layers, a distribution whose entropy
   is **indistinguishable from a random vector** (Fig. 2). The vector carries essentially no
   direct next-token signal.

2. **It fires on structural, not lexical, positions.** The tokens whose hidden states most align
   with the direction are **syntactic / role / control positions** — `<think>`, the `assistant`
   turn slot, math delimiters (in the math task); constraint and format markers (in the IF task) —
   rather than content words (App. C.1, Fig. C-heatmap).

3. **Its effect is realized through computation, not injection.** A causal *push-through* (inject
   the vector, then run the remaining layers) shows that although the logit lens is uninformative,
   the final output is pushed toward a *transition-to-answering* register (Okay/Alright). The vector
   sets a **downstream-realized computational intent**, not a token to emit.

Together: a mid/low steering vector is a **task-specific reasoning-pattern direction**. It works
because the ten-plus layers below it translate a single additive "mode" cue into the appropriate
per-position behavior. This is the sense in which "**RL changes the reasoning pattern**."

---

## X.3 Boundary: Why the Same Recipe Fails in High Layers (compressed; full analysis in App. C)

The claim above also predicts its own boundary. Near the output, a layer's residual is already close
to the logit, so its **principal computation directions coincide with the output-token directions**
(App. C.2, Fig. C-pca). A high-layer steering vector therefore increasingly spends its capacity on
**directly reshaping the next-token distribution** rather than on an internal pattern — the
reflection-word (Wait/But/…) component of the vector grows monotonically with depth and peaks at the
top layer (App. C.1). But "directly reshaping the output with one global vector" is an
**ill-posed objective**: different positions require different next tokens, so a single additive
direction faces irreconcilable per-position gradient conflict (App. C.3), which manifests exactly as
the two pathologies in Fig. 1 — norm blow-up or vanishing gradients. Full-param/LoRA escape this
because they edit **weights (computation)**, not a single global direction.

> We stress this is a matter of **degree, not kind**: a norm-controlled causal ablation shows that
> removing the token-direction component (at matched norm) still retains most of a high-layer
> vector's output effect (App. C.4). We therefore say high-layer vectors are **increasingly biased
> toward the token-output space**, not that they perform *pure* token manipulation.

---

## X.4 Interpretation: The Space RL-Acquired Capability Is Confined To

Combining the puzzle (X.1), the positive claim (X.2), and the boundary (X.3), we can characterize
the **locus and range** of the RL gain along three axes — each backed by measurement:

- **Depth.** The additive representation of the RL gain lives in the **low/mid layers**. High layers
  still carry RL signal (full-param/LoRA succeed), but there the gain does **not** exist in additive
  form — it becomes inseparable from output-token manipulation.

- **Within-layer geometry.** The direction lies in the **low-variance / output-orthogonal** subspace
  of the hidden state (logit-lens entropy = random baseline; `cos` to hidden mean ≈ 0; App. C.2).
  RL changes a direction the model does *not* use to directly write tokens — one that must be
  *computed through* by later layers.

- **Dimensionality (a canonical, near-1D direction per layer).** Vectors trained **independently**
  (different layer ranges, learning rates, seeds) for the *same* layer converge to **nearly the same
  direction**: mean pairwise cosine **0.5–0.83** (vs. ≈0 for random vectors in ℝ²⁵⁶⁰), highest in
  the mid layers (~0.82). Stacking these independent solutions and taking the SVD gives a
  **participation ratio of ≈1.3–2.0 per layer** — i.e. the per-layer RL gain occupies a
  **1–2 dimensional canonical subspace**, not a random feasible solution (App. C.6, Fig. C-consistency).

**Putting it together — the one-sentence answer.**
> The capability RLVR adds over the base model is confined to a **low-dimensional (≈1–2D per layer),
> output-orthogonal "reasoning-pattern" subspace located in the low/mid layers**. It is low-rank
> (hence recoverable by a few steering vectors — Finding 1), directionally **canonical** (independent
> runs agree), and **positionally definite** (mid/low, tail subspace). The high-layer RL signal
> exists but cannot be compressed into an additive direction — it is reachable only by editing
> weights (full-param/LoRA), not by steering.

This gives a mechanism-constrained answer to *what RL changes and where*: **not the model's direct
output mapping, but a compact, canonical set of mid-layer reasoning-computation directions.**

---

## Figure / Appendix Map

**正文图（≤2 张）**
- **Fig. 1** — Layer-wise training dynamics: downstream reward / vector-norm / grad-norm vs. step
  (the high-layer pathology that makes the X.1 puzzle concrete). ← 主图
- （可选 Fig. 2）logit-lens entropy vs. depth with random baseline（X.2 证据 1 的一眼图）。

**附录 C（承接 FINDINGS.md 的 12 条证据）**
- C.1 Representation: logit-lens entropy / top-tokens / step-evolution / token-type heatmap
      → FINDINGS 证据 1,2,3,8（figs 1,2,5,7,8）
- C.2 Geometry: vector vs. hidden PCA subspace overlap → 证据 6（fig6）+ W_U energy 证据 5（fig4）
- C.3 Optimization: two training pathologies + per-position gradient conflict → 证据 10,11（figs 11,12）
- C.4 Causal ablation: norm-controlled ΔKL, "degree not kind" → 证据 12（fig13）
- C.5 Two-model generality (math vs. IF) → 证据 9 + fig10
- C.6 Cross-run canonical-subspace consistency（新）→ mean-cos 0.5–0.83, PR≈1–2（fig-consistency 待生成）

---

## TODO / 需你补齐的硬数据
- **表 1 的 full-param / LoRA 分层分数**：正文 puzzle 的地基，需用你们的 full/LoRA 分层实验实测填 ✓/✗ → 数字。
  （现有 log：`4bopd_fullparam*.log`、`4bopd_lora8.log`、`4bopd_layers26-32*.log` 可提取。）
- **App. C.6 的一致性图**：跨-run cos 与 PR 已算出（见 README 更新），出一张 fig-consistency。
- **术语统一**：正文用 "steering vector" / 你们论文若叫 "trainable token vector" 需全局替换。

### 内部把关（正文不写）
- X.2 的 claim 是"操作性/表征性"论断：steer 有效层 = RL 增益可加性栖息层；full/LoRA 高层有效 ⇒ 高层有信号但非可加性，措辞已区分。
- X.3 已全程用程度性措辞（"increasingly biased"），并显式指向 C.4 的消融校准，避免绝对化。
- X.4 的"1–2D 典范子空间"目前由**跨-run SVD**支撑（旁证）；如需直证"能力总维度"，补 multi-vector rank 饱和扫描（N=1,2,4,8,16），脚本可用 `examples/representation/4b_opd_multi.sh`。
