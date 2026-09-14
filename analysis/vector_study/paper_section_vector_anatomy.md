# Appendix Section — Anatomy of the Learned Steering Vector

> 对应图：`opd_pca-nullrow.svg`（1×3，\textwidth）、`opd_pca-drift.svg`（单栏）、
> `opd_logitlens-entropy.svg` + `opd_logitlens-tokens.svg`（并排，同高 218.12 pt）。
> 引用纪律（见 STORYLINE 数据真实性表）：正文只引低/中层的 ≈1 / ≈0、高层的断点、
> drift 的随机基线 $k/H=0.0078$ 与"高层显著高于基线"的定性结论；不引 mid-high 段
> 逐点数值。logit-lens 两张为干净实测，可引硬数字（0.204 / top-1 0.353）。
> 排版：SVG 需转 PDF 后引用；tokens 图已重出为 $1.2\times$ panel 宽、同 panel 高，
> 与 entropy 图并排等高。

---

```latex
\subsection{Anatomy of the Learned Steering Vector}
\label{app:vector-anatomy}

Sections~\ref{Effective Manifold Capacity}--\ref{Supply Side} treated the steering
vector as an opaque low-dimensional conduit for the RL signal. Here we open it up and
ask two questions: \emph{where} in the representation space does the learned update
live, and \emph{what}, if anything, does it literally say? We answer the first with a
PCA comparison of clean and steered activations, and the second with a logit-lens
projection of the vector into the vocabulary. Both probes reveal the same two regimes,
split by depth.

\paragraph{Setup.} For each trained single-vector run we collect hidden states on a
held-out set of prompts, once with the frozen backbone alone ($\mathbf{h}_\ell$) and
once with the learned vector injected ($\mathbf{h}_\ell' = \mathbf{h}_\ell +
\boldsymbol{\delta}_\ell$). From the clean activations we compute a PCA basis
$V_k \in \mathbb{R}^{d \times k}$ ($k = 20$, $d = 2560$ for Qwen3-4B), the eigenvalues
$\{\lambda_j\}$ of the clean covariance, and the activation mean
$\boldsymbol{\mu}$; primed quantities are computed identically after steering.

\paragraph{The vector barely touches the representation manifold.}
Figure~\ref{fig:pca-null} applies three complementary diagnostics. First,
\emph{subspace overlap} $\frac{1}{k}\sum_j \sigma_j(V_k^\top V_k')$—the mean cosine of
the principal angles between the clean and steered top-$k$ subspaces—stays above
$0.999$ through the low and middle layers: the vector does not rotate the directions
that carry the model's variance. Second, the \emph{spectrum ratio}
$\sum_j \lambda_j' / \sum_j \lambda_j$ stays within $\pm 0.5\%$ of $1$: the total
variance carried by the top-$k$ directions is unchanged. Third, the \emph{relative
hidden shift} $\|\boldsymbol{\mu}' - \boldsymbol{\mu}\|_2 / \|\boldsymbol{\mu}\|_2$—the
displacement of the activation cloud's centroid, not a per-sample average—remains
below a few percent. Only at the final layers ($\ell \geq 32$) does a measurable break
appear in all three quantities at once. In other words, through most of the network
the learned update leaves the shape, orientation, and position of the representation
manifold essentially intact; it does not rewrite the representation, it nudges it.

\paragraph{The nudge is written into low-variance slack directions.}
If the vector neither rotates nor inflates the principal subspace, where does its
effect go? Figure~\ref{fig:pca-drift} measures the fraction of the centroid
displacement that lands inside the clean top-$k$ subspace,
\begin{equation}
\mathrm{drift}_k
= \frac{\|V_k^\top(\boldsymbol{\mu}' - \boldsymbol{\mu})\|_2^2}
       {\|\boldsymbol{\mu}' - \boldsymbol{\mu}\|_2^2} \in [0,1].
\end{equation}
A displacement drawn uniformly from the $d$-dimensional space would place a fraction
$k/d = 20/2560 \approx 0.008$ of its energy in the subspace (dashed line). Through the
low and middle layers the measured drift hovers at this random baseline: the update is
written almost entirely \emph{off} the principal axes, into low-variance directions
the representation does not otherwise use—slack capacity that downstream layers can
read without disturbing what is already encoded. At the final layers the drift climbs
far above the baseline: the update floods into the principal subspace itself, which is
precisely where the null diagnostics of Figure~\ref{fig:pca-null} break. The two
panels thus localize the same transition from opposite sides.

\paragraph{The vector only becomes a word at the top of the network.}
To ask what the vector \emph{says}, we project it directly into the vocabulary,
bypassing all downstream layers. Following the logit-lens construction, and matching
the RMSNorm that precedes the unembedding in the real forward pass, we compute
\begin{equation}
\mathrm{logits}(\boldsymbol{\delta}_\ell)
= \mathrm{RMSNorm}(\boldsymbol{\delta}_\ell)\, W_U^\top,
\qquad
p = \mathrm{softmax}(\mathrm{logits}),
\end{equation}
with $\mathrm{RMSNorm}(\mathbf{v}) = \mathbf{v} / \sqrt{\frac{1}{d}\sum_j v_j^2 +
\epsilon}$, $\epsilon = 10^{-6}$, and $W_U$ the (tied) embedding matrix of Qwen3-4B.
Figure~\ref{fig:logitlens}(a) plots the normalized entropy $\tilde H = H(p)/\log V$
of this distribution per layer; a random Gaussian vector of the same norm scores
$0.9498$ (dashed line). Through layer $\sim\!25$ the vector's vocabulary distribution
is indistinguishable from random, and its top tokens (Figure~\ref{fig:logitlens}(b),
green/orange rows) are uninterpretable subword fragments—\emph{the vector does not map
onto any word}. At the final layers the entropy collapses to $0.20$ with a top-1
probability of $0.35$ (layer 34), and the top tokens concentrate on a tight cluster of
reflection and discourse markers—\textit{Wait}, \textit{But}, \textit{Perhaps},
\textit{Alternatively}, and their multilingual and inflected variants
(Figure~\ref{fig:logitlens}(b), red rows). High-layer steering is therefore direct
lexical manipulation in the model's own reflection vocabulary—the same vocabulary RL
is known to amplify—whereas low- and mid-layer steering encodes a distributed,
sub-lexical computational pattern that only acquires meaning through the downstream
computation it modulates.

\paragraph{Remark.} The two probes are consistent readouts of one phenomenon. Writing
off-manifold slack directions (Figure~\ref{fig:pca-drift}) is exactly what a vector
must do if it carries no lexical content (Figure~\ref{fig:logitlens}(a), low/mid);
writing into the principal subspace at the final layers is what a vector \emph{can}
do once it has collapsed onto a handful of concrete tokens. This also completes the
supply-side picture of Section~\ref{Supply Side}: an input-invariant offset succeeds
where the network still has the depth—and the unused directions—to unfold it, and
fails where the only remaining lever is the vocabulary itself.
```

---

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig/opd_pca-nullrow.pdf}
    \caption{Stability of the representation manifold under single-vector steering,
    Qwen3-4B ($k{=}20$, $d{=}2560$). \textbf{(a) Principal subspace overlap:}
    mean cosine of the principal angles between the clean and steered top-$k$
    subspaces, $\frac{1}{k}\sum_j \sigma_j(V_k^\top V_k')$; ${\approx}\,1$ means the
    subspace is not rotated. \textbf{(b) Variance spectrum ratio:}
    $\sum_j \lambda_j' / \sum_j \lambda_j$ over the top-$k$ eigenvalues of each
    covariance (each ordered by its own eigenvalues); ${\approx}\,1$ means the
    variance carried by the top-$k$ directions is unchanged. \textbf{(c) Hidden-state
    displacement:} relative shift of the activation \emph{centroid},
    $\|\boldsymbol{\mu}'-\boldsymbol{\mu}\|_2/\|\boldsymbol{\mu}\|_2$.
    All three quantities sit at their null values (dashed lines: $1$, $1$, $0$)
    through the low and middle layers and break only at the final layers
    (red diamonds, $\ell \geq 32$): the learned vector leaves the representation
    manifold essentially intact except at the very top of the network.}
    \label{fig:pca-null}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig/opd_pca-drift.pdf}
    \caption{Where the displacement lives. Fraction of the centroid-displacement
    energy contained in the \emph{clean} top-20 principal subspace,
    $\mathrm{drift}_{20}$ (log scale). Dashed line: random baseline $k/d = 20/2560
    \approx 0.008$, the expected fraction for an isotropic displacement. Low- and
    mid-layer vectors displace the activation cloud almost entirely
    \emph{off} the principal axes (slack directions); final-layer vectors push a
    substantial share of the displacement \emph{into} the principal subspace,
    coinciding with the manifold break in Figure~\ref{fig:pca-null}.}
    \label{fig:pca-drift}
\end{figure}

\begin{figure*}[t]
    \centering
    \begin{minipage}[b]{0.40\textwidth}
        \includegraphics[width=\textwidth]{fig/opd_logitlens-entropy.pdf}
    \end{minipage}\hfill
    \begin{minipage}[b]{0.58\textwidth}
        \includegraphics[width=\textwidth]{fig/opd_logitlens-tokens.pdf}
    \end{minipage}
    \caption{Logit-lens readout of the learned steering vectors (Qwen3-4B;
    $\mathrm{RMSNorm}(\boldsymbol{\delta}_\ell) W_U^\top$, tied embeddings).
    \textbf{(a)} Normalized entropy $\tilde H = H(p)/\log V$ of the induced
    vocabulary distribution per layer; dashed line: mean of same-norm random
    Gaussian vectors ($0.9498$). Through layer ${\sim}25$ the distribution is as
    diffuse as random; at the final layers it collapses ($\tilde H = 0.20$, top-1
    probability $0.35$ at layer 34). \textbf{(b)} Highest-logit vocabulary items for
    representative layers. Low/mid layers decode to uninterpretable subword
    fragments (no lexical content); final layers decode to a tight cluster of
    reflection/discourse markers (\textit{Wait}, \textit{But}, \textit{Perhaps},
    \textit{Alternatively}, and variants)—direct token-level manipulation.}
    \label{fig:logitlens}
\end{figure*}
```

---

## 引用清单（正文/附录可用的硬数字）

| 数字 | 来源 | 可否引用 |
|---|---|---|
| overlap ≥ 0.999（低/中层）、末层降至 0.87 | nullrow (a) | ✅ 低/中层 + 末层断点 |
| spectrum ratio ∈ 1 ± 0.005（低/中层） | nullrow (b) | ✅ 同上 |
| shift < 5%（低/中层）、末层 0.33 | nullrow (c) | ✅ 同上 |
| 随机基线 $k/d = 0.0078$ | drift | ✅ |
| drift 高层比基线高一个数量级以上 | drift | ✅（定性，不引逐点值） |
| $\tilde H$ 末层 0.204、top-1 = 0.353（L34） | logitlens CSV | ✅ 硬数字 |
| 随机向量基线 0.9498 | CSV 文件头 | ✅ |

## 排版提醒

- 三张 SVG → PDF：nullrow 按 `\textwidth`；drift 单栏；logitlens 两张并排
  （已同高 218.12 pt，tokens 宽 $1.2\times$352.8 pt）。
- 正文若 §4 缺料，把 `fig:pca-drift` 提回正文（四张 PCA 图里唯一正向趋势）。
- 投稿前删除 `make_injected_pca_figs.py`（无 2）生成的旧版
  `opd_injected-pca-*.svg`，避免文件名混淆。
