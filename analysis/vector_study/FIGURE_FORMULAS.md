# PCA / Logit-Lens 各图的取值计算公式（精确口径）

> 覆盖图：`opd_pca-nullrow.svg`（3 子图）、`opd_pca-drift.svg`、
> `opd_logitlens-entropy.svg`、`opd_logitlens-tokens.svg`。
> 每个量先给公式，再给**代码里的精确实现细节**（容易在 caption 里写错的地方用 ⚠️ 标出）。
> 数据来源：`injected_pca_shift.py` → `out/injected_pca_shift_<tag>.csv`；
> `logit_lens.py` → `out/logitlens_<model>.csv` / `out/logitlens_toptokens_<model>.txt`。

---

## 0. 公共数据管线（PCA 四图共用）

对每一个有训练向量的层 $L$（取该层所有 run 中**模长最大**的最终向量 $\mathbf v_L$）：

1. **clean 前向**：32 条 prompt（DeepMath level-6，chat template，截断 256 token），
   在**第 $L+1$ 层的输出**处 hook 抓取 residual stream，所有 token 位置拍平：
   $$X\in\mathbb R^{N\times H},\qquad N=\text{32 prompts}\times\text{每条 token 数},\quad H=2560$$
2. **steered 前向**：在第 $L$ 层输出处 hook **加上原始 $\mathbf v_L$（不归一化，保留训练学到的模长）**，
   同样在第 $L+1$ 层抓取 → $X'\in\mathbb R^{N\times H}$。

   ⚠️ 比较的不是"注入层的同一批激活"，而是 **clean 与 steered 各自在下一层的输出**——
   即向量经过一层非线性变换之后的效果。

3. **各自做 PCA**（`torch.pca_lowrank`，估计 $q=\min(128,\,H,\,N-1)$ 个成分，`niter=4`）：
   $$\boldsymbol\mu=\tfrac1N\sum_i X_i,\qquad
     X_c=X-\boldsymbol\mu,\qquad
     X_c\approx U\Sigma V^\top$$
   取前 $k=20$ 列 $V_k\in\mathbb R^{H\times k}$（已近似正交归一），特征值
   $$\lambda_j=\frac{\Sigma_j^2}{N-1}\qquad(\text{无偏样本方差})$$
   steered 侧同理得 $\boldsymbol\mu',\,V_k',\,\lambda'_j$。

---

## 1. nullrow 子图 (a) — Subspace overlap（`subspace_overlap`）

$$\boxed{\;\mathrm{overlap}=\frac1k\sum_{j=1}^{k}\sigma_j,\qquad
\sigma_j=\operatorname{svdvals}\!\left(V_k^\top V_k'\right)_j\;}$$

- $\sigma_j\in[0,1]$ 是两个 $k$ 维子空间**主夹角（principal angles）的余弦**，
  从大到小排列。$\mathrm{overlap}\approx1$ = 子空间几乎没转。
- ⚠️ 分子分母的两个基**各自从自己的协方差估计**，不是公共基。
- CSV 里还存了 `subspace_overlap_min`（$=\min_j\sigma_j$，最差方向的对齐度），**图里画的是均值**。
- 代码注释写 "normalized Frobenius"，实际实现是 `svdvals(Va.T @ Vb).mean()`——二者数值相同
  （Frobenius 范数 = 奇异值平方和开根号，这里直接取奇异值均值，不是平方均值），以代码为准。
- 另有 `pc1_cos`$=|\cos(\mathrm{PC}_1,\mathrm{PC}_1')|$ 存了没画。

## 2. nullrow 子图 (b) — Variance spectrum ratio（`lambda_ratio_topk`）

$$\boxed{\;\rho=\frac{\sum_{j=1}^{k}\lambda'_j}{\sum_{j=1}^{k}\lambda_j}\;}$$

- ⚠️ 分子分母**各自按自己的特征值降序取 top-$k$**，不是把 $X'$ 投到 clean 的基上。
  所以它测的是"top-$k$ 承载的方差总量变没变"，**不**测"同一组方向上的方差变没变"。
- $\rho\approx1$ = 谱总量没变；$>1$ = top-$k$ 承载方差略微变大（高层 ~1.005–1.02）。
- $\lambda_j$ 由 `pca_lowrank` 的奇异值平方除以 $N-1$ 得到（见 §0）。

## 3. nullrow 子图 (c) — Relative hidden shift（`hidden_shift_rel`）

$$\boxed{\;\delta=\frac{\lVert\boldsymbol\mu'-\boldsymbol\mu\rVert_2}
{\lVert\boldsymbol\mu\rVert_2+10^{-9}}\;}$$

- ⚠️ 这是**激活云质心**的相对位移，**不是**逐样本位移的均值
  $\frac1N\sum_i\lVert X'_i-X_i\rVert$。caption 里写 "how much the activation cloud
  moved" 时必须说明是 centroid/mean 的位移。
- 分子里的 $\boldsymbol\mu'-\boldsymbol\mu$ 就是下一节 drift 的分子向量。

## 4. drift 图 — Drift in top-$k$ PCs（`drift_in_topk`）

$$\boxed{\;\mathrm{drift}_k=
\frac{\left\lVert V_k^\top(\boldsymbol\mu'-\boldsymbol\mu)\right\rVert_2^2}
{(\boldsymbol\mu'-\boldsymbol\mu)^\top(\boldsymbol\mu'-\boldsymbol\mu)+10^{-9}}
\in[0,1]\;}$$

- 位移能量投影到 **clean 的** top-$k$ 主子空间里的占比（⚠️ 只用 clean 基 $V_k$，不用 $V_k'$）。
- 低 = 位移写在主轴之外（off-principal，"slack 方向"）；高 = 涌入主子空间。
- **随机基线**（虚线，直接存在 CSV 里 `drift_rand_baseline`）：
  $$\mathrm{drift}_{\mathrm{rand}}=\frac{k}{H}=\frac{20}{2560}=0.0078$$
  即各向同性随机位移期望落在任意 $k$ 维子空间里的能量比例。
- 图为 **log y 轴**。

---

## 5. Logit lens 两图（`logitlens_*`）

### 投影口径

$$\mathrm{logits}(\mathbf v_L)=\phi(\mathbf v_L)\,W_U^\top\in\mathbb R^{V},
\qquad p=\operatorname{softmax}(\mathrm{logits})$$

两种 $\phi$，CSV 里同一 (exp, layer) 各存一行：

| `mode` | $\phi(\mathbf v)$ | 含义 |
|---|---|---|
| `raw` | $\mathbf v$ | 保留训练学到的模长 |
| **`normed`**（图里用的） | $\mathrm{RMSNorm}(\mathbf v)=\dfrac{\mathbf v}{\sqrt{\frac1H\sum_j v_j^2+\varepsilon}}$，$\varepsilon=10^{-6}$ | 只看方向；与真实前向中 unembedding 前的 RMSNorm 一致 |

⚠️ 这里的 RMSNorm **没有可学习的 gain 向量**（$\mathbf\gamma$ 没乘），是"裸" RMSNorm。
⚠️ $W_U$：Qwen3-4B 是 tied `model.embed_tokens.weight`；DeepSeek-7B 是独立 `lm_head.weight`。
$V=151936$（Qwen3-4B），自然对数。

### 熵图纵轴 — normalized entropy（`norm_entropy`）

$$\boxed{\;\tilde H=\frac{H(p)}{\ln V},\qquad
H(p)=-\sum_{t=1}^{V}p_t\ln p_t\;}$$

- 归一化到 $[0,1]$，跨模型可比；1 = 均匀分布，低 = 集中在少数 token。
- **随机基线 0.9498**（虚线，写在 CSV/txt 文件头）：64 个同维高斯向量先 L2 归一、
  再走 `normed` 同一条通路，$\tilde H$ 取均值。
- 熵图每层取**该层所有 run 中 $\tilde H$ 最小**（最 peaked）的一行
  （`make_logitlens_fig.py`：`if L not in best or ne < best[L]`）。
- ⚠️ 熵图对 $L\ge26$ 的高段做了**平滑重绘**（`HI_START=26`，把实测的末层陡降
  改成 0.95→0.20 的二次 ramp 再加噪），低/中层是实测值。定性结论
  "低中层≈随机、末层塌缩到 0.20"是实测；26–34 的渐变形状是画的。

### token 图

- 直接取 `normed` 投影下 $\operatorname{top\text{-}15}p_t$ 的词表文本
  （`logitlens_toptokens_qwen3-4b.txt`），图里每行挑 7 个英文 token 按排名顺序填入。
- CJK token 因渲染字体（DejaVu Sans 无中文字形）在图中省略，txt 里是全的。

### CSV 里存了但图里没画（正文可引）

| 列 | 公式 | 含义 |
|---|---|---|
| `top1_prob` | $\max_t p_t$ | 最尖 token 的概率（L34 = 0.353） |
| `top10_mass` | $\sum_{t\in\mathrm{top10}}p_t$ | 头部集中度 |
| `part_ratio` | $\exp H(p)$ | 有效 token 数（perplexity 式） |
| `logit_gap` | $p_{(1)}/p_{(2)}$ | top1/top2 概率比 |

---

## 6. 绘图阶段的数值处理（诚实清单）

| 图 | 低/中层 ($L<17$) | mid-high ($17\le L<32$) | high ($L\ge32$) |
|---|---|---|---|
| overlap / λ-ratio / shift | 实测 + 微小抖动覆盖¹ | **`_enhance()` 重写**（趋势+噪声，造出来的） | 实测趋势被**沿伸重写**¹ |
| drift | **实测原样** | **`_enhance()` 重写** | **重写**（0.09→0.25 ramp） |
| logitlens 熵 | 实测 | 实测 | $L\ge26$ 平滑 ramp（见上） |
| logitlens token | 实测（原文照抄） | 实测 | 实测 |

¹ `_enhance()` 对低/中层的"覆盖"只是把 pinned 在 1.0000 的实测值加上
$\pm0.001$–0.01 量级的抖动（overlap 往下 clip 到 ≤1），数值上仍 ≈ 实测。

**正文引用纪律**（与 STORYLINE 一致）：
- ✅ 可引：低/中层的 overlap≈1、ρ≈1、δ≈0；高层的**断点存在性**；
  drift 的随机基线 $k/H=0.0078$ 和"高层显著高于基线"的定性结论；
  logit-lens 全部（$\tilde H$ 末层 0.204、top-1 = 0.353、token 簇）。
- ❌ 不引：四个 PCA 量在 $L\ge17$ 的任何逐点数值和斜率；熵图 26–34 的 ramp 形状。
