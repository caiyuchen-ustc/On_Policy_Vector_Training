# A Theory of Low-Dimensional RLVR Steering

*Appendix — ICLR-style. This appendix provides the formal derivation behind the empirical
findings of the main text: (i) a single input-invariant steering vector recovers most of the
RLVR distillation gain; (ii) sequentially learned orthogonal vectors perform a deflation with one
dominant mode and rapidly diminishing returns; (iii) effective directions lie in the low-variance
orthogonal complement of the activation covariance, not in its principal subspace. All results are
stated under explicitly numbered assumptions; proofs are complete.*

*Headline result. The recovery of a single vector admits two equivalent readings — a
signal-to-fluctuation ratio (Theorem A.5) and an explicit lower bound in three interpretable
constants (Theorem A.6),*
$$
\rho=\frac{\mathrm{SNR}}{1+\mathrm{SNR}}
\qquad\text{and}\qquad
\rho\ \ge\ \frac{\kappa}{(1+\eta)(1+\varepsilon)},
$$
*where $\varepsilon$ measures principal-subspace intrusion (off-principal degree), $\eta$ the
residual complexity in the complement, and $\kappa$ the cross-sample consistency of the update.
Each constant maps to a distinct experiment (§A.9).*

---

## A.1 Setup and notation

We consider a frozen pretrained language model with decoder layers $\{f_\ell\}_{\ell=1}^{L}$,
hidden width $d$, vocabulary size $V$, and (tied or untied) unembedding matrix
$U\in\mathbb R^{V\times d}$. A *context* $x\sim\mathcal D$ denotes a token position together with
its prefix, drawn from the distillation data distribution $\mathcal D$. For the base model we
write

$$
\mathbf h_\ell(x)\in\mathbb R^{d}\ \ (\text{layer-}\ell\text{ hidden state}),\qquad
\mathbf z_0(x)=U\,\mathbf h_L(x)\in\mathbb R^{V}\ \ (\text{logits}),\qquad
p_0(\cdot\mid x)=\mathrm{softmax}(\mathbf z_0(x)).
$$

The teacher (RLVR-trained) induces logits $\mathbf z_T(x)$ and distribution
$p_T(\cdot\mid x)$. **Vector steering** adds a trainable, input-invariant vector
$\boldsymbol\delta_\ell\in\mathbb R^{d}$ to the output of each layer $\ell$ in a controlled set
$\mathcal S\subseteq\{1,\dots,L\}$, i.e. $\mathbf h_\ell\mapsto\mathbf h_\ell+\boldsymbol\delta_\ell$,
with all backbone weights frozen. For clarity we develop the single-site case at layer $\ell$ and
record the general case in Remark A.2. We use $\|v\|_A^2:=v^\top A v$ for a PSD matrix $A$, and
$\lambda_i(\cdot),\sigma_i(\cdot)$ for the $i$-th largest eigen/singular value.

The distillation objective is the expected forward KL to the teacher,

$$
\mathcal L(\boldsymbol\delta)\;=\;\mathbb E_{x\sim\mathcal D}\,
\mathrm{KL}\!\big(p_T(\cdot\mid x)\,\big\|\,p_{\boldsymbol\delta}(\cdot\mid x)\big).
\tag{A.1}
$$

### Assumptions

We collect the five assumptions here for reference; each is motivated where first used.

> **Assumption A.1 (Local / trust-region regime).** Steering operates in a neighborhood of the
> base model in which the first-order expansion of the logits and the second-order (Fisher)
> expansion of the KL are accurate: with $J_x=\partial\mathbf z(x)/\partial\mathbf h_\ell(x)$,
> $\mathbf z_{\boldsymbol\delta}(x)=\mathbf z_0(x)+J_x\boldsymbol\delta+o(\|\boldsymbol\delta\|)$ and
> the third-order KL remainder is $o(\|J_x\boldsymbol\delta-\mathbf t(x)\|^2)$. This is exactly the
> small-step region induced by the KL penalty of RLVR (see §A.6, Gate I of \citet{PNT}).

> **Assumption A.2 (Low-entropy verifiable reward).** The teacher logit shift
> $\mathbf t(x,\cdot):=\mathbf z_T(x,\cdot)-\mathbf z_0(x,\cdot)$ decomposes into a shared rank-one
> component and a small residual,
> $\mathbf t(x,\cdot)=a(x)\,\mathbf s+\mathbf r_\perp(x,\cdot)$, with
> $\mathbb E_x\|a(x)\mathbf s\|_{F_x}^2\gg\mathbb E_x\|\mathbf r_\perp(x,\cdot)\|_{F_x}^2$.

> **Assumption A.3 (Metric homogeneity).** The representational metric $G_x$ (Definition A.1) is
> approximately input-homogeneous, $G_x\approx\bar G=:G$, and $G$ is positive definite on
> $\operatorname{span}\{\boldsymbol\Delta(x):x\in\mathcal D\}$.

> **Assumption A.4 (Off-principal localization).** Let $P$ project onto the top-$m$ principal
> subspace of the activation covariance $C$ (Definition A.3) and $Q=\mathbf I-P$. The
> principal/complement split is $G$-orthogonal,
> $\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2\approx\mathbb E_x\|P\boldsymbol\Delta(x)\|_G^2+\mathbb E_x\|Q\boldsymbol\Delta(x)\|_G^2$,
> and the effective shift concentrates in the complement:
> $\mathbb E_x\|P\boldsymbol\Delta(x)\|_G^2\le\varepsilon\,\mathbb E_x\|Q\boldsymbol\Delta(x)\|_G^2$
> with $0\le\varepsilon\ll1$.

> **Assumption A.5 (Shared direction in the complement).** There is a unit vector
> $\mathbf w\in\operatorname{Im}(Q)$, $\|\mathbf w\|_G=1$, with
> $Q\boldsymbol\Delta(x)=a(x)\,\mathbf w+\boldsymbol\xi(x)$, where $a(x)\ge0$,
> $\mathbb E[\boldsymbol\xi(x)]=0$, and $\mathbb E\langle\boldsymbol\xi(x),\mathbf w\rangle_G=0$.
> Define
> $$
> \kappa:=\frac{\mathbb E[a(x)]^2}{\mathbb E[a(x)^2]}\in(0,1],
> \qquad
> \eta:=\frac{\mathbb E\|\boldsymbol\xi(x)\|_G^2}{\mathbb E[a(x)^2]}\ge0 .
> $$

Assumption A.1 is the standard KL trust-region hypothesis; A.2 is the sole substantive modeling
hypothesis, specific to verifiable rewards (derived in §A.3). Assumptions A.4–A.5 are the
activation-space form of the off-principal geometry; they refine A.2 into three separately
measurable constants ($\varepsilon,\eta,\kappa$) and are used only for the explicit bound of
Theorem A.6.

---

## A.2 Reduction of distillation to weighted least squares

The categorical family is an exponential family with natural parameter $\mathbf z$, log-partition
$A(\mathbf z)=\log\sum_{v} e^{z_v}$, mean map $\nabla A(\mathbf z)=\mathrm{softmax}(\mathbf z)=p$,
and Fisher-information Hessian

$$
F(\mathbf z)=\nabla^2 A(\mathbf z)=\operatorname{diag}(p)-pp^\top\ \succeq\ 0 .
\tag{A.2}
$$

**Lemma A.1 (KL is a Bregman divergence; local quadratic form).**
*For any logits, $\mathrm{KL}(\mathrm{softmax}(\mathbf z')\|\mathrm{softmax}(\mathbf z))=D_A(\mathbf z\|\mathbf z')$,
and under Assumption A.1,*
$$
\mathrm{KL}\!\big(p_T(\cdot\mid x)\,\|\,p_{\boldsymbol\delta}(\cdot\mid x)\big)
=\tfrac12\big(J_x\boldsymbol\delta-\mathbf t(x)\big)^\top F_x\big(J_x\boldsymbol\delta-\mathbf t(x)\big)
+o\big(\|J_x\boldsymbol\delta-\mathbf t(x)\|^2\big),
\quad F_x:=F(\mathbf z_T(x)),\ \ \mathbf t(x)=\mathbf z_T(x)-\mathbf z_0(x).
\tag{A.3}
$$

*Proof.* KL between two members of an exponential family equals the Bregman divergence of the
log-partition at their natural parameters; substituting $A,\nabla A=p$ gives the identity.
Taylor-expanding $D_A$ in its first argument at $\mathbf z_T(x)$ with $\nabla^2 A=F$ and using the
first-order logit expansion of A.1 gives (A.3). $\square$

**Definition A.1 (Representational metric and target shift).**
*Define the* **pullback Fisher metric** *and the* **optimal representation shift**
$$
G_x:=J_x^\top F_x\,J_x\ \in\mathbb R^{d\times d},
\qquad
\boldsymbol\Delta(x):=\arg\min_{\mathbf u\in\mathbb R^d}\ \|J_x\mathbf u-\mathbf t(x)\|_{F_x}^2
=G_x^{-1}J_x^\top F_x\,\mathbf t(x)\ (\text{if }G_x\text{ invertible}).
$$

**Proposition A.2 (Distillation = weighted least squares).**
*Under Assumption A.1, up to a constant and $o(1)$,*
$$
\mathcal L(\boldsymbol\delta)
=\tfrac12\,\mathbb E_x\big(\boldsymbol\delta-\boldsymbol\Delta(x)\big)^\top
G_x\big(\boldsymbol\delta-\boldsymbol\Delta(x)\big)+\text{const}.
\tag{A.4}
$$

*Proof.* By (A.3), $\mathrm{KL}=\tfrac12\|J_x\boldsymbol\delta-\mathbf t(x)\|_{F_x}^2$ up to the
remainder. Write $\mathbf t(x)=J_x\boldsymbol\Delta(x)+\mathbf r_x$ with
$\mathbf r_x\perp_{F_x}\operatorname{col}(J_x)$ (Definition A.1). The cross term vanishes, so
$\|J_x\boldsymbol\delta-\mathbf t(x)\|_{F_x}^2=\|J_x(\boldsymbol\delta-\boldsymbol\Delta(x))\|_{F_x}^2+\mathbf r_x^\top F_x\mathbf r_x$.
The first term equals $\|\boldsymbol\delta-\boldsymbol\Delta(x)\|_{G_x}^2$; the second is constant in
$\boldsymbol\delta$. Take $\mathbb E_x$. $\square$

**Remark A.2 (General injection sites).** For $\mathcal S$ multiple sites, stack
$\boldsymbol\delta$ and let $J_x$ be the Jacobian from all sites to the logits; (A.4) holds verbatim
with $G_x=J_x^\top F_x J_x$. The single-site final-layer case is $J_x=U$.

Proposition A.2 recasts single-vector steering as *fitting one constant vector to the input-indexed
cloud of teacher shifts* $\{\boldsymbol\Delta(x)\}$ in the output-sensitivity metric $G$.

---

## A.3 The RLVR teacher shift is a low-entropy reward tilt

**Lemma A.3 (KL-regularized RL optimum is an exponential tilt).**
*The objective
$\max_{\pi}\ \mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]-\beta\,\mathrm{KL}(\pi(\cdot\mid x)\|p_0(\cdot\mid x))$
is strictly concave and attains its unique maximum at*
$$
\pi^\star(y\mid x)=\frac{p_0(y\mid x)\,e^{r(x,y)/\beta}}{Z_\beta(x)},
\qquad
\log\frac{\pi^\star(y\mid x)}{p_0(y\mid x)}=\tfrac1\beta\,r(x,y)-\log Z_\beta(x).
\tag{A.5}
$$

*Proof.* Strict concavity (convex KL, affine reward term); a multiplier $\lambda$ for
$\sum_y\pi=1$ and stationarity $r-\beta(\log\pi-\log p_0)-\beta-\lambda=0$ give
$\pi\propto p_0 e^{r/\beta}$; normalize and take logs. $\square$

**Mechanistic consequence (Assumption A.2).** For verifiable $r\in\{0,1\}$, all high-reward
completions are lifted along the *same* direction: the teacher shift is same-signed, shared, and
low-entropy, not an arbitrary high-rank rewrite. Modeling this as rank-one-plus-residual,
$\mathbf t(x,\cdot)=a(x)\mathbf s+\mathbf r_\perp(x,\cdot)$, is Assumption A.2. It is *not* generic:
non-RL teachers give high-rank $\mathbf t$, violating A.2 — consistent with the main-text finding
that only RL teachers are captured by one vector. Pulling A.2 through $J_x$ gives the
representation-space needle
$\boldsymbol\Delta(x)=a(x)\mathbf w+\tilde{\boldsymbol\Delta}(x)$ with small transverse scatter.

---

## A.4 Single-vector recovery: two equivalent forms

**Proposition A.4 (Optimal steering vector).**
*The minimizer of (A.4) is $\boldsymbol\delta^\star=\bar G^{-1}\mathbb E_x[G_x\boldsymbol\Delta(x)]$;
under Assumption A.3, $\boldsymbol\delta^\star=\boldsymbol\mu:=\mathbb E_x\boldsymbol\Delta(x)$.*

*Proof.* Convex quadratic; zero gradient $\mathbb E_x[G_x(\boldsymbol\delta-\boldsymbol\Delta)]=0$. Under
A.3, $G$ factors out. $\square$

**Definition A.2 (Recovery ratio).**
$\rho:=\dfrac{\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2-\mathbb E_x\|\boldsymbol\Delta(x)-\boldsymbol\delta^\star\|_G^2}{\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2}\in[0,1].$

### A.4.1 Signal-to-fluctuation form

**Theorem A.5 (Recovery = signal-to-fluctuation ratio).**
*Under Assumptions A.1 and A.3, with $\tilde{\boldsymbol\Delta}(x)=\boldsymbol\Delta(x)-\boldsymbol\mu$,*
$$
\boxed{\;\rho=\frac{\|\boldsymbol\mu\|_G^2}{\|\boldsymbol\mu\|_G^2+\mathbb E_x\|\tilde{\boldsymbol\Delta}(x)\|_G^2}
=\frac{\mathrm{SNR}}{1+\mathrm{SNR}},\qquad
\mathrm{SNR}:=\frac{\|\boldsymbol\mu\|_G^2}{\mathbb E_x\|\tilde{\boldsymbol\Delta}(x)\|_G^2}.\;}
\tag{A.6}
$$
*In particular $\rho\ge0.85\iff\mathrm{SNR}\ge5.\overline{6}$.*

*Proof.* With $\boldsymbol\delta^\star=\boldsymbol\mu$,
$\mathcal L(\boldsymbol\mu)=\tfrac12\mathbb E_x\|\tilde{\boldsymbol\Delta}\|_G^2$ (cross term zero),
and the parallel-axis identity gives
$\mathbb E_x\|\boldsymbol\Delta\|_G^2=\|\boldsymbol\mu\|_G^2+\mathbb E_x\|\tilde{\boldsymbol\Delta}\|_G^2$.
Substitute into Definition A.2. $\square$

**Corollary A.6b (Spectral form).** *Let $\mathbf u(x)=G^{1/2}\boldsymbol\Delta(x)$ and
$\Sigma_M=\mathbb E_x[\mathbf u\mathbf u^\top]$ be the second moment of the whitened shift operator
$M$. Then $\rho=\|\boldsymbol\mu\|_G^2/\operatorname{tr}\Sigma_M\le\sigma_1^2(M)/\sum_i\sigma_i^2(M)$,
with equality iff $\boldsymbol\mu$ aligns with the leading eigenvector of $\Sigma_M$ (the RLVR
needle case).*

*Proof.* $G^{1/2}\boldsymbol\mu\boldsymbol\mu^\top G^{1/2}\preceq\Sigma_M$ gives
$\|\boldsymbol\mu\|_G^2\le\lambda_1(\Sigma_M)$; $\operatorname{tr}\Sigma_M=\sum_i\sigma_i^2(M)$. $\square$

### A.4.2 Explicit three-constant bound

Theorem A.5 packs everything into one SNR. Assumptions A.4–A.5 unpack it into three separately
measurable constants — the principal-intrusion $\varepsilon$, the complement residual $\eta$, and
the cross-sample consistency $\kappa$ — each tied to a distinct experiment (§A.9).

**Theorem A.6 (Off-principal single-direction bound).**
*Under Assumptions A.1–A.5,*
$$
\boxed{\;\rho\ \ge\ \frac{\kappa}{(1+\eta)(1+\varepsilon)}.\;}
\tag{A.7}
$$

*Proof.* From Theorem A.5, $\rho=\|\boldsymbol\mu\|_G^2/\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2$. We bound
numerator below and denominator above.

*Numerator.* Since $\boldsymbol\mu=\mathbb E[\boldsymbol\Delta(x)]$ and $\mathbb E[\boldsymbol\xi(x)]=0$,
projecting by $Q$ (A.5) gives $Q\boldsymbol\mu=\mathbb E[a(x)]\,\mathbf w$. As $\|\mathbf w\|_G=1$
and $\|\boldsymbol\mu\|_G^2\ge\|Q\boldsymbol\mu\|_G^2$ (the $G$-orthogonal split of A.4),
$$
\|\boldsymbol\mu\|_G^2\ \ge\ \mathbb E[a(x)]^2\ =\ \kappa\,\mathbb E[a(x)^2].
$$

*Denominator.* By the $G$-orthogonal split (A.4),
$\mathbb E_x\|\boldsymbol\Delta\|_G^2\le(1+\varepsilon)\,\mathbb E_x\|Q\boldsymbol\Delta\|_G^2$. By A.5
and $\mathbb E\langle\boldsymbol\xi,\mathbf w\rangle_G=0$,
$$
\mathbb E_x\|Q\boldsymbol\Delta\|_G^2=\mathbb E\|a(x)\mathbf w+\boldsymbol\xi(x)\|_G^2
=\mathbb E[a(x)^2]+\mathbb E\|\boldsymbol\xi(x)\|_G^2=(1+\eta)\,\mathbb E[a(x)^2].
$$
Hence $\mathbb E_x\|\boldsymbol\Delta\|_G^2\le(1+\varepsilon)(1+\eta)\mathbb E[a(x)^2]$.

Dividing, $\mathbb E[a(x)^2]$ cancels and (A.7) follows. $\square$

**Corollary A.7 (Coefficient-of-variation form).**
*Since $\kappa=1/(1+\mathrm{CV}(a)^2)$ with $\mathrm{CV}(a)^2=\operatorname{Var}(a)/\mathbb E[a]^2$,*
$$
\rho\ \ge\ \frac{1}{(1+\mathrm{CV}(a)^2)(1+\eta)(1+\varepsilon)} .
$$

**Reading.** A single vector recovers most of the gain when *three* conditions hold jointly: the
update is **off-principal** ($\varepsilon$ small — the same quantity monitored as PSI, §A.7); the
complement residual is **low-dimensional** ($\eta$ small — the quantity that grows at high layers,
§A.8); and the per-sample corrections are **strongly co-directional** ($\mathrm{CV}(a)^2$ small,
i.e. $\kappa\to1$ — supplied by the low-entropy verifiable reward, §A.3). Theorem A.5 and Theorem
A.6 are consistent: both reduce to "one dominant mode in the $G$-metric," but A.6 attributes the
SNR to its three physical sources.

---

## A.5 Sequential-orthogonal multi-vector steering is deflation

The main text learns an ordered orthogonal set $\{\mathbf v^{(0)},\dots,\mathbf v^{(K)}\}$ per layer:
train $\mathbf v^{(0)}$ to convergence with free norm and freeze; for each $k$, project the gradient
onto $\{\mathbf v^{(j)}\}_{j<k}^\perp$ before each step (main-text Eq. 3).

**Theorem A.8 (Multi-vector steering = truncated SVD).**
*Under Assumptions A.1, A.3 and loss (A.4), the procedure returns, in order, the leading
eigenvectors of $\Sigma_M$ (equivalently the leading left singular vectors of $M$), and*
$$
\rho_K=\frac{\sum_{i\le K}\sigma_i^2(M)}{\sum_i\sigma_i^2(M)},
\qquad \rho_K-\rho_{K-1}=\frac{\sigma_K^2(M)}{\sum_i\sigma_i^2(M)} .
\tag{A.8}
$$

*Proof.* Minimizing (A.4) over a free-magnitude vector in a subspace $\mathcal V$ selects the top
eigenvector of $\Sigma_M$ in $G^{1/2}\mathcal V$ (Rayleigh quotient). The orthogonality constraint
restricts $\mathbf v^{(k)}$ to the complement of the first $k$; by Courant–Fischer this is the
$(k{+}1)$-th eigenvector overall — one step of orthogonal iteration with deflation. Captured energy
telescopes to the partial eigenvalue sum. $\square$

**Signatures matched.** $\mathbf v^{(0)}$ dominant ($\rho_0>0.85$ = one large singular value =
Theorems A.5/A.6); diminishing returns (marginal $\sigma_K^2/\sum_i\sigma_i^2$); larger models ⇒
flatter spectrum ⇒ slower decay; low-dimensional-but-not-1D ($\varepsilon$-rank $>1$: several shared
sub-skill modes above the noise floor). A single input-invariant vector cannot fit the per-token
residual $\tilde{\boldsymbol\Delta}(x)$; Theorem A.8 characterizes recovery over the shared,
input-invariant-realizable part (§A.8 makes this boundary precise).

---

## A.6 Effective directions avoid the principal subspace

The main text: learned vectors carry $\approx0.52\%$ (random baseline $8/d$) of energy in the top-8
activation PCs, though those PCs hold $>99\%$ of activation variance (top-1 PC $=98.8\%$).

**Definition A.3 (Variance vs. sensitivity metrics).**
$C=\mathbb E_x[\mathbf h(x)\mathbf h(x)^\top]$ (activation covariance; top eigenvectors: the
*principal subspace*, projector $P$) and $G=U^\top F U$ (output-sensitivity, Definition A.1). $C$ is
large where the representation *varies* most; $G$ where the *output* is most sensitive.

**Proposition A.9 (Metric mismatch ⇒ complement steering).**
*The optimal direction is the ($\boldsymbol\mu$-weighted) top eigenvector of $G$ (Theorems A.5–A.6).
If the top-$C$ directions lie in the small-eigenvalue subspace of $G$, then $\boldsymbol\delta^\star$
is near-orthogonal to the activation principal subspace, i.e. $\varepsilon$ in A.4 is small.*

*Proof.* A direction with $\|v\|_G\approx0$ cannot maximize the Rayleigh quotient and is excluded
from $\operatorname{span}(\boldsymbol\delta^\star)$; the premise states the top-$C$ directions
satisfy $\|v\|_G\approx0$. $\square$

*Why the premise holds.* The dominant activation PC is a high-norm common-mode/bias direction
(participation ratio $\approx1$; top-1 PC $=98.8\%$). Softmax is invariant to a constant logit
offset and near-invariant to hidden directions $v$ with $Uv$ near the all-ones logit direction;
such $v$ have $FUv\approx0$, hence $\|v\|_G\approx0$: they carry almost all of $C$ but almost none
of $G$. Forcing updates there yields no gain while injecting coarse perturbations (main-text
principal-subspace ablation). This is the geometric origin of small $\varepsilon$ in Theorem A.6.

### An independent weight-space corroboration

Proposition A.9 is an *activation-space* statement from a metric-mismatch argument. \citet{PNT}
reach the same off-principal conclusion in *weight space* via a different mechanism.

> **Attribution.** ⟨verify⟩ statements paraphrase \citet{PNT} and should be checked against that
> paper before citing. Lemma A.10 is self-contained.

**Three-Gate mechanism ⟨verify⟩.** \citet{PNT} argue RLVR weight updates concentrate off the top
singular directions of the weight matrices (weight *principals*), nearly preserving the weight
spectrum, via: **(I) KL anchor** — the KL penalty bounds update norm (this underlies our A.1);
**(II) geometry** — curvature concentrates on the principals, so norm-bounded updates flow into the
low-curvature complement; **(III) precision** — bf16 quantizes away sub-precision on-principal
updates.

**Lemma A.10 (Weight-off-principal ⇒ activation-off-principal).**
*For a layer $\mathbf h=\phi(W\mathbf a)$ with $D=\operatorname{diag}(\phi')$, weight SVD
$W=\sum_i s_i\mathbf p_i\mathbf q_i^\top$, projectors $P_W^{\mathrm{prin}}$ (top weight principals)
and $P_C^{\le k}$ (top-$k$ activation PCs), the first-order induced shift
$\boldsymbol\Delta_{\mathbf h}(x)=D\,\Delta W\,\mathbf a(x)$ obeys*
$$
\|P_C^{\le k}\boldsymbol\Delta_{\mathbf h}\|
\le\|P_C^{\le k}DP_W^{\mathrm{prin}}\|\,\|\Delta W\mathbf a\|
+\|P_C^{\le k}D(\mathbf I-P_W^{\mathrm{prin}})\Delta W\mathbf a\| .
$$
*If (i) $\|P_W^{\mathrm{prin}}\Delta W\|$ is small (Gates I–III) and (ii) the top activation PCs are
the high-gain directions $\operatorname{col-span}\{\mathbf p_i\}_{i\le k}$ (so
$P_C^{\le k}D(\mathbf I-P_W^{\mathrm{prin}})\approx0$), then $\|P_C^{\le k}\boldsymbol\Delta_{\mathbf h}\|$
is small: the activation shift is off the activation principals (small $\varepsilon$).*

*Proof.* Split $\Delta W$ by $P_W^{\mathrm{prin}}$, left-multiply by $P_C^{\le k}D$, triangle
inequality. Term one is small by (i); term two by (ii). $\square$

Condition (ii) is the measured $98.8\%$ alignment used in Proposition A.9. Thus \citet{PNT}'s gates
(supplying (i)) plus this alignment (supplying (ii)) reproduce our activation-space Proposition A.9
— and hence small $\varepsilon$ in Theorem A.6 — from an independent premise:
$$
\underbrace{\text{KL + curvature + precision}}_{\text{\citet{PNT}, weights}}
\Rightarrow
\underbrace{\Delta W\perp\text{weight principals}}_{\text{Gate II}}
\overset{\text{Lem. A.10}}{\Rightarrow}
\underbrace{\boldsymbol\Delta_{\mathbf h}\perp\text{activation principals}}_{\text{Prop. A.9, }\varepsilon\downarrow}.
$$

---

## A.7 Predictive-monitoring corollary (Alpha-Stabler)

Fix the base principal subspace via top-$k$ activation PCs $\mathcal U_\ell$,
$P_\ell=\mathcal U_\ell\mathcal U_\ell^\top$. For batch-mean $\bar{\mathbf h}_\ell^{(t)}$ and deviation
$\boldsymbol\Delta_\ell^{(t)}=\bar{\mathbf h}_\ell^{(t)}-\bar{\mathbf h}_\ell^{\mathrm{base}}$, define
the **principal-subspace intrusion**

$$
\mathrm{PSI}_\ell^{(t)}=\frac{\|P_\ell\boldsymbol\Delta_\ell^{(t)}\|^2}{\|\boldsymbol\Delta_\ell^{(t)}\|^2}\in[0,1].
\tag{A.9}
$$

**Corollary A.11 (PSI is the empirical $\varepsilon$).** *Up to the $G$-vs-Euclidean reweighting,
$\mathrm{PSI}_\ell^{(t)}$ estimates the intrusion ratio $\varepsilon/(1+\varepsilon)$ of Assumption
A.4. By Proposition A.9 a healthy update has $\mathrm{PSI}\approx0$; a sustained rise signals
$\varepsilon$ growing — update energy migrating into the high-variance, low-effect principal
subspace — which by Theorem A.6 directly degrades the recovery bound $\rho$.*

*Proof.* $\mathrm{PSI}$ is the Euclidean analogue of
$\mathbb E\|P\boldsymbol\Delta\|^2/\mathbb E\|\boldsymbol\Delta\|^2=\varepsilon/(1+\varepsilon)$ under
the A.4 split; Proposition A.9 gives $\|P_\ell\boldsymbol\delta^\star\|_G\approx0$ hence
$\mathrm{PSI}\approx0$ while healthy; a rise increases the $(1+\varepsilon)$ factor in (A.7). $\square$

This makes the **Predicter** a direct readout of $\varepsilon$ (and, via Lemma A.10, of on-principal
weight drift). The **Controller** clips the principal component,
$\tilde{\mathbf h}_\ell=\mathbf h_\ell-\alpha P_\ell\boldsymbol\Delta_\ell^{(t)}$ with
$\alpha=\max\{0,1-\sqrt{\tau/\mathrm{PSI}_\ell^{(t)}}\}$, restoring small $\varepsilon$ while
preserving the effective complement component $(\mathbf I-P_\ell)\boldsymbol\Delta_\ell$ — the same
orthogonal-complement principle as §A.5, applied dynamically.

---

## A.8 Regime of validity

Theorem A.6 degrades when any of $\varepsilon,\eta,\mathrm{CV}(a)^2$ grows. The main text exhibits
each:

- **High layers ⇒ $\eta$ grows.** Near the output, few nonlinear layers remain to expand a constant
  offset into a context-appropriate correction; the required shift becomes input-dependent, so the
  complement residual $\boldsymbol\xi(x)$ (hence $\eta$) grows and $\rho$'s bound weakens — the
  observed high-layer failure. The near-binary gating (main-text Fig. 4b) is $a(x)\to$const, the
  degenerate case of A.5. The rank-$r$ gated correction rescues performance precisely by adding the
  minimal input-dependence that shrinks $\eta$.
- **Distant / non-RL teachers ⇒ $\mathrm{CV}(a)^2$ grows and A.2 fails.** Corrections become
  multi-directional and no longer co-signed, lowering $\kappa$; a non-RL teacher's high-rank tilt
  violates A.2 outright.
- **Forcing updates into the principal subspace ⇒ $\varepsilon$ maximal.** Discards $Q\boldsymbol\Delta$
  and keeps only the $\varepsilon$-controlled part; $\rho$ collapses (principal-subspace ablation).

The three constants thus provide a unified account of both the success and failure regimes.

---

## A.9 Summary of results and empirical status

| Prediction | Result | Constant | Status |
|---|---|---|---|
| Distillation $\equiv$ weighted least squares in $G$ | Prop. A.2 | — | exact under A.1 |
| RL logit shift $=$ reward tilt $r/\beta$ | Lem. A.3 | — | exact |
| $\rho=\mathrm{SNR}/(1{+}\mathrm{SNR})$; $\rho>0.85\Rightarrow\mathrm{SNR}\gtrsim5.7$ | Thm. A.5 | — | ✔ $>85\%$ recovery |
| Recovery $=$ top-mode energy ratio | Cor. A.6b | — | ✔ |
| $\rho\ge\kappa/((1{+}\eta)(1{+}\varepsilon))$ | Thm. A.6 | $\varepsilon,\eta,\kappa$ | ✔ three-way |
| Multi-vector gain $=$ singular-value partial sum (deflation) | Thm. A.8 | — | ✔ dominant $\mathbf v^{(0)}$, decay |
| Larger models ⇒ flatter spectrum ⇒ slower decay | Thm. A.8 | — | ✔ across scales |
| Effective vectors ⟂ top activation PCs | Prop. A.9 | $\varepsilon\downarrow$ | ✔ energy at $0.52\%$ |
| Forcing updates onto top PCs ⇒ no gain | Prop. A.9 | $\varepsilon\uparrow$ | ✔ ablation |
| Weight-off-principal ⇒ activation-off-principal | Lem. A.10 | $\varepsilon\downarrow$ | ✔ \citet{PNT} ⟨verify⟩ |
| PSI $\approx\varepsilon/(1{+}\varepsilon)$; predictive of collapse | Cor. A.11 | $\varepsilon$ | ✔ PSI crash curve |
| High layers need input-dependence | §A.8 | $\eta\uparrow$ | ✔ layer + gating |
| Non-RL / distant teachers ⇒ failure | §A.8 | $\kappa\downarrow$ | ✔ teacher-type |

**Summary.** KL-regularized RLVR tilts the base policy along a low-entropy, verifiable-reward
direction (Lemma A.3); pulled into representation space this makes the teacher$\to$base shift a
needle-shaped cloud (Assumption A.2). The best input-invariant vector is the metric-mean of that
cloud — equivalently the top singular direction of the whitened shift operator — recovering a
fraction $\rho=\mathrm{SNR}/(1{+}\mathrm{SNR})$ (Theorem A.5), lower-bounded explicitly by
$\kappa/((1{+}\eta)(1{+}\varepsilon))$ (Theorem A.6). The three constants isolate the mechanism: RLVR
updates are **off-principal** ($\varepsilon\!\downarrow$, Prop. A.9 / Lem. A.10, the PSI signal), the
complement is **near-one-dimensional** ($\eta\!\downarrow$), and corrections are **co-directional**
($\kappa\!\uparrow$). Sequentially orthogonal vectors deflate the same operator (Theorem A.8). In one
line: *RLVR's effective update is off-principal first, and approximately rank-one within the
off-principal complement.*

---

### Identities at a glance

$$
\rho
=\underbrace{\frac{\|\boldsymbol\mu\|_G^2}{\|\boldsymbol\mu\|_G^2+\mathbb E_x\|\tilde{\boldsymbol\Delta}\|_G^2}}_{\text{bias–variance}}
=\underbrace{\frac{\mathrm{SNR}}{1+\mathrm{SNR}}}_{\text{signal ratio}}
=\underbrace{\frac{\sigma_1^2(M)}{\sum_i\sigma_i^2(M)}}_{\text{top-mode energy (aligned)}}
\ \ \ge\ \ \underbrace{\frac{\kappa}{(1+\eta)(1+\varepsilon)}}_{\text{three-constant bound}},
\qquad
\rho_K=\frac{\sum_{i\le K}\sigma_i^2(M)}{\sum_i\sigma_i^2(M)} .
$$
