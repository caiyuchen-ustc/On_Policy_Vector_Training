<!--
AUTHOR NOTE (remove before submission)
======================================
This is a paper-ready draft of the Alpha-Stabler section. Statements containing
[TO FILL] require real experimental results. The current local PSI plotting
scripts contain constructed or partially constructed trajectories and should
not be cited as quantitative evidence until replaced by PSI measured from
complete, single-run training logs. In particular, do not report collapse
prediction or controller improvements without real multi-seed experiments.
-->

# Alpha-Stabler: Geometry-Guided Prediction and Stabilization

The preceding analyses identify a common geometric signature of effective RL
updates. Under stable training, the additional activation signal remains
concentrated in the low-variance complement of the base model's activation
principal subspace. In contrast, performance collapse is accompanied by a
rapid and increasingly volatile migration of update energy into the
variance-dominant principal directions. This observation suggests that the same
geometry used to interpret RL-induced capability changes can also be used to
monitor and control the training process.

We operationalize this idea through **Alpha-Stabler**, a plug-and-play
framework with two components. The **Predictor** monitors
principal-subspace intrusion (PSI) and raises an early warning when the update
geometry departs persistently from its stable regime. The **Controller**
attenuates only the intrusive principal component while preserving the
low-variance complement associated with effective capability acquisition.
Alpha-Stabler does not modify the optimizer, reward function, or policy
objective, and can reuse the frozen reference-model forward pass already
required by KL-regularized RL.

## 1. Fixed reference geometry

Let \(f_{\theta_0}\) denote the frozen base model and
\(\mathcal{S}\) the set of monitored layers. For each
\(\ell\in\mathcal{S}\), we collect centered token-level activations of the base
model on a calibration set \(\mathcal{D}_{\mathrm{cal}}\):

\[
\mathbf{C}_{\ell}
=
\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{cal}}}
\left[
    \left(\mathbf{h}_{\ell}^{0}(x)-\boldsymbol{\mu}_{\ell}^{0}\right)
    \left(\mathbf{h}_{\ell}^{0}(x)-\boldsymbol{\mu}_{\ell}^{0}\right)^{\top}
\right],
\]

where \(\boldsymbol{\mu}_{\ell}^{0}\) is the corresponding activation mean.
Let

\[
\mathbf{U}_{\ell}
=
\left[
    \mathbf{u}_{\ell,1},
    \ldots,
    \mathbf{u}_{\ell,r}
\right]
\in\mathbb{R}^{d\times r},
\qquad
r=\left\lceil qd\right\rceil,
\]

contain the top \(r\) eigenvectors of \(\mathbf{C}_{\ell}\). We use
\(q=0.10\) by default and define the fixed projectors

\[
\mathbf{P}_{\ell}
=
\mathbf{U}_{\ell}\mathbf{U}_{\ell}^{\top},
\qquad
\mathbf{Q}_{\ell}
=
\mathbf{I}-\mathbf{P}_{\ell}.
\]

The bases are computed once from the base model and remain fixed throughout
training. Fixing the reference geometry is important: recomputing the PCs from
the evolving policy would allow the monitored subspace to move with the
instability and would obscure the quantity that Alpha-Stabler is intended to
detect.

## 2. Online principal-subspace intrusion

At training step \(t\), let
\(\overline{\mathbf{h}}_{\ell}^{(t)}\) denote the batch-mean activation of the
current policy and let
\(\overline{\mathbf{h}}_{\ell}^{0,(t)}\) denote the activation of the frozen
base model on the same batch. We define the observed activation shift as

\[
\widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
=
\overline{\mathbf{h}}_{\ell}^{(t)}
-
\overline{\mathbf{h}}_{\ell}^{0,(t)}.
\tag{8}
\]

The layer-wise principal-subspace intrusion is

\[
\operatorname{PSI}_{\ell}^{(t)}
=
\frac{
    \left\|
        \mathbf{P}_{\ell}
        \widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
    \right\|_2^2
}{
    \left\|
        \widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
    \right\|_2^2+\epsilon
}
\in[0,1],
\tag{9}
\]

where \(\epsilon>0\) prevents numerical instability when the observed shift is
small. For reporting a single global statistic, we use an energy-weighted
aggregation:

\[
\operatorname{PSI}^{(t)}
=
\frac{
    \sum_{\ell\in\mathcal{S}}
    \left\|
        \mathbf{P}_{\ell}
        \widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
    \right\|_2^2
}{
    \sum_{\ell\in\mathcal{S}}
    \left\|
        \widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
    \right\|_2^2+\epsilon
}.
\tag{10}
\]

This aggregation gives less weight to layers whose measured shift is near
zero. A low PSI indicates that the RL-induced change remains predominantly in
the low-variance complement, whereas increasing PSI indicates that a larger
fraction of the update has entered the variance-dominant principal subspace.

For an isotropic random direction, the expected projection energy is
\(r/d\approx10\%\). This value is a dimension-matched geometric reference, not
the intervention threshold. In practice, stable RL updates exhibit a
model-specific PSI baseline substantially below this random reference;
Alpha-Stabler therefore calibrates its thresholds from the observed warm-up
regime rather than fixing them at \(10\%\).

## 3. Predictor: detecting departure from the stable regime

Absolute PSI values vary across layers, models, and data distributions. We
therefore estimate a robust baseline during a warm-up period of
\(T_{\mathrm{warm}}\) monitoring steps. For each layer, let

\[
m_{\ell}
=
\operatorname{median}_{t\leq T_{\mathrm{warm}}}
\operatorname{PSI}_{\ell}^{(t)},
\]

\[
s_{\ell}
=
1.4826\,
\operatorname{median}_{t\leq T_{\mathrm{warm}}}
\left|
    \operatorname{PSI}_{\ell}^{(t)}-m_{\ell}
\right|
+\epsilon
\tag{11}
\]

be the median and robust scale estimate. The layer-specific warning threshold
is

\[
\tau_{\ell}^{\mathrm{on}}
=
m_{\ell}+c_{\mathrm{on}}s_{\ell},
\tag{12}
\]

where \(c_{\mathrm{on}}\) controls sensitivity. We smooth the online signal
using an exponential moving average,

\[
\overline{\psi}_{\ell}^{(t)}
=
\beta\overline{\psi}_{\ell}^{(t-1)}
+
(1-\beta)\operatorname{PSI}_{\ell}^{(t)}.
\tag{13}
\]

To reduce false positives caused by isolated spikes, the Predictor combines a
level test with a persistence test. Define

\[
z_{\ell}^{(t)}
=
\frac{
    \overline{\psi}_{\ell}^{(t)}-m_{\ell}
}{
    s_{\ell}
},
\qquad
Z^{(t)}
=
\max_{\ell\in\mathcal{S}}z_{\ell}^{(t)}.
\tag{14}
\]

An alert is raised when \(Z^{(t)}\geq c_{\mathrm{on}}\) for
\(p\) consecutive monitoring steps. Optionally, a positive-slope condition can
be added:

\[
g^{(t)}
=
\frac{
    \operatorname{PSI}^{(t)}
    -
    \operatorname{PSI}^{(t-w)}
}{w}
\geq g_{\min}.
\tag{15}
\]

The persistence and slope conditions distinguish sustained geometric drift
from ordinary minibatch variation. The Predictor can be used independently as
an early-warning monitor, or jointly with the Controller described next.

## 4. Controller: clipping only the intrusive component

The Controller acts at layers whose smoothed PSI exceeds the calibrated safe
threshold. Decompose the observed shift into principal and complementary
components:

\[
\widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
=
\underbrace{
    \mathbf{P}_{\ell}
    \widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
}_{\text{principal component}}
+
\underbrace{
    \mathbf{Q}_{\ell}
    \widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
}_{\text{complement component}}.
\tag{16}
\]

For a target intrusion level \(\tau_{\ell}\in(0,1)\), define

\[
\gamma_{\ell}^{(t)}
=
\begin{cases}
1,
&
\operatorname{PSI}_{\ell}^{(t)}
\leq\tau_{\ell},
\\[6pt]
\sqrt{
\dfrac{
    \tau_{\ell}
    \left(1-\operatorname{PSI}_{\ell}^{(t)}\right)
}{
    \left(1-\tau_{\ell}\right)
    \operatorname{PSI}_{\ell}^{(t)}
    +\epsilon
}
},
&
\operatorname{PSI}_{\ell}^{(t)}
>\tau_{\ell}.
\end{cases}
\tag{17}
\]

The controlled hidden state for every token \(i\) in the batch is

\[
\widetilde{\mathbf{h}}_{\ell,i}^{(t)}
=
\mathbf{h}_{\ell,i}^{(t)}
-
\lambda
\left(
    1-\gamma_{\ell}^{(t)}
\right)
\operatorname{sg}
\left[
    \mathbf{P}_{\ell}
    \widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
\right],
\tag{18}
\]

where \(\operatorname{sg}[\cdot]\) denotes stop-gradient and
\(\lambda\in[0,1]\) controls intervention strength. With
\(\lambda=1\), the principal component is rescaled by
\(\gamma_{\ell}^{(t)}\), while the complementary component is unchanged:

\[
\widetilde{\boldsymbol{\Delta}}_{\ell}^{(t)}
=
\gamma_{\ell}^{(t)}
\mathbf{P}_{\ell}
\widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}
+
\mathbf{Q}_{\ell}
\widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}.
\tag{19}
\]

Substituting Equation (17) into Equation (19) gives

\[
\frac{
    \left\|
        \mathbf{P}_{\ell}
        \widetilde{\boldsymbol{\Delta}}_{\ell}^{(t)}
    \right\|_2^2
}{
    \left\|
        \widetilde{\boldsymbol{\Delta}}_{\ell}^{(t)}
    \right\|_2^2
}
=
\tau_{\ell}
\tag{20}
\]

whenever the Controller is active and \(\lambda=1\). Thus, the intervention
caps the intrusion ratio at the calibrated safe level rather than removing the
entire principal component. Crucially,

\[
\mathbf{Q}_{\ell}
\widetilde{\boldsymbol{\Delta}}_{\ell}^{(t)}
=
\mathbf{Q}_{\ell}
\widehat{\boldsymbol{\Delta}}_{\ell}^{(t)},
\tag{21}
\]

so the low-variance complement identified as recovery-bearing in the preceding
sections is preserved exactly.

The correction is shared across tokens within the minibatch. It therefore
removes only the batch-level intrusive component and leaves token-dependent
structure intact. The stop-gradient operator prevents the correction itself
from introducing cross-example gradient terms; it changes the forward signal
and consequently the policy gradient, but requires no modification to the
optimizer.

### Hysteresis and activation policy

To prevent rapid switching near the threshold, we use separate activation and
deactivation thresholds:

\[
\tau_{\ell}^{\mathrm{off}}
=
m_{\ell}+c_{\mathrm{off}}s_{\ell},
\qquad
c_{\mathrm{off}}<c_{\mathrm{on}}.
\tag{22}
\]

The Controller activates after the Predictor has remained above
\(\tau_{\ell}^{\mathrm{on}}\) for \(p\) checks and deactivates only after PSI
remains below \(\tau_{\ell}^{\mathrm{off}}\) for \(p_{\mathrm{off}}\) checks.
This hysteresis avoids reacting to isolated minibatch noise.

## 5. Alpha-Stabler algorithm

The following algorithm summarizes calibration, prediction, and control.

```latex
\begin{algorithm}[t]
\caption{Alpha-Stabler: online prediction and control of principal-subspace intrusion}
\label{alg:alpha_stabler}
\begin{algorithmic}[1]
\Require
Policy $f_{\theta}$; frozen base model $f_{\theta_0}$;
monitored layers $\mathcal{S}$; PC fraction $q$;
warm-up length $T_{\mathrm{warm}}$; monitoring interval $M$;
EMA coefficient $\beta$; thresholds $c_{\mathrm{on}},c_{\mathrm{off}}$;
patience $p,p_{\mathrm{off}}$; controller strength $\lambda$.

\State Compute the top $r=\lceil qd\rceil$ base-model PCs
$\mathbf U_{\ell}$ and projectors
$\mathbf P_{\ell}=\mathbf U_{\ell}\mathbf U_{\ell}^{\top}$
for every $\ell\in\mathcal S$.

\State Run $T_{\mathrm{warm}}$ monitoring steps without intervention.
\State Estimate the robust baseline $(m_{\ell},s_{\ell})$ using
Equation~\eqref{eq:psi_baseline}.
\State Set
$\tau_{\ell}^{\mathrm{on}}=m_{\ell}+c_{\mathrm{on}}s_{\ell}$ and
$\tau_{\ell}^{\mathrm{off}}=m_{\ell}+c_{\mathrm{off}}s_{\ell}$.
\State $\mathrm{active}\gets\mathrm{False}$;
$n_{\mathrm{on}}\gets0$; $n_{\mathrm{off}}\gets0$.

\For{training step $t=1,2,\ldots$}
    \State Sample an RL minibatch $\mathcal B_t$.
    \State Obtain base activations
    $\{\mathbf h_{\ell}^{0,(t)}\}_{\ell\in\mathcal S}$
    from the frozen reference forward pass.
    \State Run the current policy to the monitored layers and compute
    $\widehat{\boldsymbol\Delta}_{\ell}^{(t)}$ and
    $\operatorname{PSI}_{\ell}^{(t)}$ using
    Equations~\eqref{eq:activation_shift}--\eqref{eq:psi}.
    \State Update the EMA $\overline{\psi}_{\ell}^{(t)}$.

    \If{$t \bmod M = 0$}
        \If{the level/persistence test is positive}
            \State $n_{\mathrm{on}}\gets n_{\mathrm{on}}+1$
        \Else
            \State $n_{\mathrm{on}}\gets0$
        \EndIf
        \If{$n_{\mathrm{on}}\ge p$}
            \State Emit a collapse warning;
            $\mathrm{active}\gets\mathrm{True}$
        \EndIf
    \EndIf

    \If{$\mathrm{active}$}
        \For{$\ell\in\mathcal S$}
            \State Compute $\gamma_{\ell}^{(t)}$ using
            Equation~\eqref{eq:controller_gain}.
            \State Apply the controlled activation
            $\widetilde{\mathbf h}_{\ell}^{(t)}$ using
            Equation~\eqref{eq:controller_update}.
        \EndFor
        \If{$\overline{\psi}_{\ell}^{(t)}
        <\tau_{\ell}^{\mathrm{off}}$ for all monitored layers}
            \State $n_{\mathrm{off}}\gets n_{\mathrm{off}}+1$
        \Else
            \State $n_{\mathrm{off}}\gets0$
        \EndIf
        \If{$n_{\mathrm{off}}\ge p_{\mathrm{off}}$}
            \State $\mathrm{active}\gets\mathrm{False}$
        \EndIf
    \EndIf

    \State Compute the original RL objective using the controlled forward pass.
    \State Update $\theta$ with the original optimizer.
\EndFor
\end{algorithmic}
\end{algorithm}
```

For direct inclusion in LaTeX, the equation labels in the algorithm should be
renamed to match the labels used in the final manuscript. A compact
implementation can execute monitoring every \(M>1\) steps while retaining the
previous controller state between checks.

## 6. Computational overhead

The PCA bases are computed once before RL training. At each monitored step, the
additional geometric computation at layer \(\ell\) consists of projecting one
batch-mean vector onto an \(r\)-dimensional basis, with cost
\(\mathcal{O}(dr)\). The correction is then broadcast across tokens. The total
online geometric cost is

\[
\mathcal{O}\!\left(
    |\mathcal{S}|dr
\right)
\]

per monitoring step, excluding the reference-model forward pass. In
KL-regularized RL, the base/reference model is already evaluated, and its
hidden states can be captured through forward hooks with little additional
compute. If no reference forward is otherwise available, we evaluate two
approximations in the ablation study: cached base activation means and
monitoring on a fixed probe minibatch every \(M\) training steps.

Memory overhead is dominated by storing the PC bases:

\[
\mathcal{O}\!\left(
    |\mathcal{S}|dr
\right).
\]

For a \(10\%\) subspace, the bases can be stored in bfloat16 or compressed with
randomized PCA. We report both wall-clock and peak-memory overhead in the
experiments.

## 7. Experimental evaluation

We evaluate Alpha-Stabler around four questions:

1. **Prediction:** Does PSI warn of collapse earlier and more reliably than
   standard optimization statistics?
2. **Stabilization:** Does the Controller reduce collapse frequency and
   preserve the score attained before instability?
3. **Specificity:** Is the benefit specific to controlling the activation
   principal subspace rather than clipping an arbitrary subspace?
4. **Robustness:** How sensitive is the method to the PC rank, monitored
   layers, threshold, monitoring interval, and controller strength?

### 7.1 Experimental setup

We use the same RL training configurations as in the preceding geometric
analysis. The diagnostic experiments include a Qwen2.5-1.5B-DeepSeek model on
Science and a Qwen2.5-7B-DeepSeek model on instruction following. The
Controller evaluation should additionally cover at least:

- two model scales;
- two task domains;
- both full-parameter and LoRA training;
- at least \([{\rm TO\ FILL}]\) independent seeds per configuration;
- both stable and collapse-prone hyperparameter regimes.

Unless otherwise specified, we use the top \(10\%\) base-model activation PCs,
monitor layers \([{\rm TO\ FILL}]\), estimate the baseline over the first
\([{\rm TO\ FILL}]\) steps, set
\(\beta=[{\rm TO\ FILL}]\),
\(c_{\mathrm{on}}=[{\rm TO\ FILL}]\),
\(c_{\mathrm{off}}=[{\rm TO\ FILL}]\),
\(p=[{\rm TO\ FILL}]\), and
\(\lambda=[{\rm TO\ FILL}]\).

### 7.2 Collapse definition and prediction metrics

Let \(\bar{s}^{(t)}\) be the EMA-smoothed validation score and
\(s_{\max}^{(t)}=\max_{u\le t}\bar{s}^{(u)}\). We define the collapse time as

\[
t_{\mathrm{collapse}}
=
\min
\left\{
    t:
    \bar{s}^{(t)}
    \leq
    s_{\max}^{(t)}-\delta_s
    \ \text{for }H_s\text{ consecutive evaluations}
\right\}.
\tag{23}
\]

The exact values of \(\delta_s\) and \(H_s\) are fixed before evaluating the
predictors. For each signal, we report:

- AUROC and AUPRC for predicting whether a run will collapse within the next
  \(H_{\mathrm{pred}}\) steps;
- median lead time
  \(t_{\mathrm{collapse}}-t_{\mathrm{alarm}}\);
- false alarms per 1,000 training steps;
- detection rate at a fixed false-positive rate;
- cross-task threshold transfer without retuning.

We compare PSI against:

- policy entropy;
- KL divergence to the reference policy;
- gradient norm;
- update norm;
- reward/score slope;
- loss curvature or optimizer second-moment statistics, when available.

### 7.3 Stabilization baselines

We compare the Alpha-Stabler Controller with:

1. **No intervention:** the original RL training configuration.
2. **Global gradient clipping:** standard clipping at the tuned norm threshold.
3. **Adaptive KL:** increasing the KL coefficient when divergence exceeds a
   threshold.
4. **Activation-norm clipping:** clipping the total activation shift without
   using its geometry.
5. **Random-subspace controller:** applying the same controller to a random
   \(r\)-dimensional subspace.
6. **Bottom-PC controller:** clipping an equal-dimensional low-variance
   subspace rather than the top-PC subspace.
7. **Always-on complement projection:** removing the principal component at
   every step, without PSI-triggered activation.
8. **Rollback/early stopping:** reverting to the latest stable checkpoint after
   the warning.

These controls distinguish the proposed geometric mechanism from generic
regularization, reduced update magnitude, and early termination.

### 7.4 Main prediction result

Figure~\(\ref{fig:psi_prediction}\) should report complete, single-run measured
trajectories rather than stitched or constructed curves. Across the evaluated
collapse-prone configurations, PSI should be measured from saved hidden states
at every monitoring step.

```latex
\begin{table}[t]
\centering
\caption{Collapse prediction performance. Lead time is measured relative to
the first sustained score collapse. All values are averaged over independent
training seeds.}
\label{tab:psi_prediction}
\begin{tabular}{lcccc}
\toprule
Signal & AUROC $\uparrow$ & AUPRC $\uparrow$
& Lead time $\uparrow$ & False alarms $\downarrow$ \\
\midrule
Score slope       & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
Policy entropy    & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
KL divergence     & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
Gradient norm     & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
PSI (ours)        & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
\bottomrule
\end{tabular}
\end{table}
```

Paper-ready result text after the measurements are available:

> Across \([{\rm TO\ FILL}]\) collapse-prone runs, PSI rises
> \([{\rm TO\ FILL}]\) steps before the first sustained score drop and achieves
> an AUROC of \([{\rm TO\ FILL}]\), outperforming entropy, KL divergence, and
> gradient norm. Thresholds calibrated on the warm-up phase transfer across
> \([{\rm TO\ FILL}]\) models/tasks with \([{\rm TO\ FILL}]\) false alarms per
> 1,000 steps.

### 7.5 Main stabilization result

The primary Controller experiment should deliberately include a
collapse-prone training regime and report multiple seeds. We recommend
reporting collapse rate, final score, best score, score AUC, maximum PSI, and
training overhead.

```latex
\begin{table*}[t]
\centering
\caption{Training stabilization. Alpha-Stabler is compared with generic
optimization and geometry-matched controls.}
\label{tab:alpha_stabler_main}
\begin{tabular}{llcccccc}
\toprule
Model / task & Method
& Collapse rate $\downarrow$
& Final score $\uparrow$
& Best score $\uparrow$
& Score AUC $\uparrow$
& Peak PSI $\downarrow$
& Overhead $\downarrow$ \\
\midrule
1.5B / Science
& No intervention & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & -- \\
& Gradient clipping & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
& Adaptive KL & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
& Random-subspace control & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
& Alpha-Stabler & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
\midrule
7B / Instruction Following
& No intervention & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & -- \\
& Gradient clipping & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
& Adaptive KL & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
& Random-subspace control & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
& Alpha-Stabler & [TO FILL] & [TO FILL] & [TO FILL]
& [TO FILL] & [TO FILL] & [TO FILL] \\
\bottomrule
\end{tabular}
\end{table*}
```

The central claim should only be made after this table is populated:

> Alpha-Stabler reduces collapse frequency while preserving the performance
> reached before intervention. Unlike global clipping and adaptive KL, it
> selectively suppresses the high-variance principal component and leaves the
> effective complementary update unchanged.

## 8. Ablation studies

### 8.1 Principal-subspace size

We vary \(q\in\{0.01,0.05,0.10,0.20,0.30\}\). Small \(q\) may omit intrusive
directions, while large \(q\) may remove useful complementary structure. We
report prediction AUROC, final score, peak PSI, and intervention frequency.

Expected interpretation:

- prediction should be weak when \(q\) is too small;
- aggressive control may reduce performance when \(q\) is too large;
- an intermediate subspace, expected near \(q=0.10\), should provide the best
  stability--performance trade-off.

### 8.2 Threshold sensitivity

We vary \(c_{\mathrm{on}}\in\{1.5,2,3,4,5\}\) and report:

- collapse rate;
- false-warning rate;
- median intervention step;
- fraction of training steps under control;
- final score.

A low threshold should trigger frequent interventions and may suppress useful
learning. A high threshold should preserve healthy training but react too late.

### 8.3 Controller strength

We vary \(\lambda\in\{0.25,0.5,0.75,1.0\}\) in Equation (18). This ablation
tests whether exact capping is necessary or whether partial attenuation is
sufficient. We additionally compare the exact-ratio gain in Equation (17) with
the simpler norm-capping approximation

\[
\gamma_{\ell,\mathrm{approx}}^{(t)}
=
\min
\left\{
    1,
    \sqrt{
        \frac{\tau_{\ell}}
        {\operatorname{PSI}_{\ell}^{(t)}+\epsilon}
    }
\right\}.
\tag{24}
\]

### 8.4 Monitored layers

We compare:

- lower layers only;
- middle layers only;
- upper layers only;
- low-to-middle layers;
- all layers.

This experiment tests whether collapse is first visible in a localized depth
range and whether controlling all layers is necessary. Layer-specific onset
times should also be reported.

### 8.5 Monitoring interval

We vary \(M\in\{1,5,10,20,50\}\). Larger intervals reduce overhead but may
decrease warning lead time. The result should report wall-clock overhead
against detection delay and collapse rate.

### 8.6 Predictor components

We compare:

- PSI level only;
- PSI slope only;
- level plus persistence;
- level plus slope plus persistence;
- fixed global threshold;
- robust warm-up-calibrated threshold.

This ablation establishes whether the proposed persistence and normalization
rules improve transfer across runs.

### 8.7 Geometric specificity

To test whether stabilization depends specifically on the activation principal
subspace, we apply the same controller to:

- a random \(r\)-dimensional subspace;
- the bottom-\(r\) activation PCs;
- the top-\(r\) PCs of a different layer;
- a PC basis estimated from shuffled or mismatched data.

If Alpha-Stabler's mechanism is correct, these controls should not match the
performance of the correctly aligned top-PC controller.

### 8.8 Reference estimation

We compare three ways to estimate
\(\widehat{\boldsymbol{\Delta}}_{\ell}^{(t)}\):

1. same-batch activations from the frozen base model;
2. cached base activation means;
3. a fixed held-out probe minibatch evaluated every \(M\) steps.

The same-batch reference is the most accurate but potentially most expensive.
The alternatives quantify the trade-off between overhead and diagnostic
quality.

### 8.9 Batch-mean versus token-level PSI

The default method controls the shared batch-mean shift. As an ablation, we
compute

\[
\operatorname{PSI}_{\mathrm{token}}^{(t)}
=
\mathbb{E}_{x,i}
\left[
\frac{
    \left\|
        \mathbf P_{\ell}
        \Delta\mathbf h_{\ell,i}^{(t)}(x)
    \right\|_2^2
}{
    \left\|
        \Delta\mathbf h_{\ell,i}^{(t)}(x)
    \right\|_2^2+\epsilon
}
\right].
\tag{25}
\]

This tests whether the collapse signal is primarily a shared drift or a
token-dependent phenomenon.

## 9. Recommended ablation table

```latex
\begin{table}[t]
\centering
\caption{Ablations of Alpha-Stabler.}
\label{tab:alpha_stabler_ablation}
\begin{tabular}{lcccc}
\toprule
Variant & Collapse rate $\downarrow$ & Final score $\uparrow$
& Peak PSI $\downarrow$ & Active steps $\downarrow$ \\
\midrule
Full Alpha-Stabler             & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
No persistence test            & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
No slope test                  & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
Fixed threshold                & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
Random subspace                & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
Bottom-PC subspace             & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
All-step control               & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
Cached base mean               & [TO FILL] & [TO FILL] & [TO FILL] & [TO FILL] \\
\bottomrule
\end{tabular}
\end{table}
```

## 10. Suggested figures

### Figure: Predictor

- two or more real collapse trajectories;
- score on one axis and PSI on the other;
- vertical lines for PSI warning and score-defined collapse;
- matched stable runs to demonstrate a low false-positive rate;
- an aggregate lead-time distribution over seeds.

Suggested caption:

```latex
\caption{
\textbf{PSI provides an early geometric warning of RL training collapse.}
\textbf{(a--b)} Score and PSI for representative collapse-prone runs.
Vertical lines mark the first PSI warning and the first sustained score
collapse. \textbf{(c)} Warning lead time across models, tasks, and seeds.
\textbf{(d)} AUROC comparison against entropy, KL divergence, gradient norm,
and score slope.
}
```

### Figure: Controller

- score curves with and without Alpha-Stabler;
- PSI curves with and without Alpha-Stabler;
- intervention strength \(\alpha_{\ell}^{(t)}\) over time;
- final-score versus peak-PSI scatter plot;
- controller overhead.

Suggested caption:

```latex
\caption{
\textbf{Alpha-Stabler suppresses principal-subspace intrusion and stabilizes
RL training.}
\textbf{(a)} Score trajectories with and without intervention.
\textbf{(b)} Corresponding PSI trajectories.
\textbf{(c)} Layer-wise controller activation over training.
\textbf{(d)} Stability--performance trade-off across thresholds and controller
strengths.
}
```

## 11. Discussion and limitations

Alpha-Stabler relies on a fixed base-model principal subspace and is therefore
best suited to training regimes in which the model remains within a local
neighborhood of its initialization. Under very large distribution shifts, the
base PCs may become stale, and periodic recalibration may be necessary.
However, recalibration must be performed cautiously: allowing the monitored
subspace to track the unstable model too quickly could hide the very drift PSI
is intended to detect.

The current Controller acts on the shared batch-mean activation shift. It may
not capture failures driven by strongly token-dependent or trajectory-local
instabilities. The token-level ablation in Equation (25) tests this boundary.
Likewise, the method assumes that the low-variance complement contains the
useful control signal, as supported by the steering and principal-subspace
interventions in the preceding sections. This assumption should be revalidated
when moving to substantially different architectures, reward models, or
multimodal settings.

Finally, PSI should be interpreted as a diagnostic geometric correlate rather
than a proven unique cause of collapse. The strongest empirical validation
requires complete real trajectories, multiple seeds, explicit warning-time
evaluation, and controlled stabilization experiments. Subject to these tests,
Alpha-Stabler closes the loop between geometric interpretation and practical
RL control: it detects when the update leaves the effective low-variance
regime and selectively suppresses the component associated with instability.

## 12. Concise main-paper version

If space is limited, the method can be summarized in the main paper as follows,
with the detailed calibration and ablations moved to the appendix:

> **Alpha-Stabler.** We fix the top-\(10\%\) activation-PC subspace of the base
> model and monitor the fraction of the RL-induced activation shift that enters
> this subspace. This principal-subspace intrusion (PSI) remains low during
> stable training but rises before or during collapse. Alpha-Stabler first
> calibrates a model-specific PSI baseline over a short warm-up period. Its
> Predictor raises a warning when PSI persistently exceeds the baseline band.
> Its Controller then rescales only the intrusive principal component to a safe
> level while preserving the complementary activation shift exactly. The
> method requires no change to the RL objective or optimizer and can reuse the
> frozen reference-model forward pass used for KL regularization.

