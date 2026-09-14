<!--
Compact main-paper draft designed for approximately two A4 pages including
one method figure and one compact result/ablation table.
Replace [TO FILL] only with measurements from complete real training runs.
-->

# Alpha-Stabler: Geometry-Guided RL Stabilization

The preceding results suggest a direct stabilization principle. Effective RL
control signals remain concentrated in the low-variance complement; increasing
their principal-subspace alignment provides little functional benefit; and
naturally occurring principal-subspace intrusion accompanies training
collapse. These observations motivate monitoring whether the RL-induced
activation change leaves its effective off-principal regime and intervening
only when such geometric drift becomes persistent.

Based on this principle, we introduce **Alpha-Stabler**, which closes the loop
between geometric diagnosis and online control. Its **Predictor** converts PSI
into an early-warning signal by identifying sustained deviations from the
model's stable geometric regime. Once a warning is issued, its **Controller**
selectively removes the excessive principal-subspace component while preserving
the heterogeneous off-principal changes associated with effective learning.
Alpha-Stabler leaves the RL objective, optimizer, and reward function unchanged.

## Method

### Online geometric monitoring

We use the token-level PSI introduced in
Equation~\ref{eq:training_psi} as the online diagnostic signal. Recall that
$X_\ell^{(t)}\in\mathbb{R}^{N\times d}$ stacks the valid token-level activation
shifts between the current policy and the frozen base model, while
$U_{\ell,r}\in\mathbb{R}^{d\times r}$ contains the fixed principal directions
of the base model at layer $\ell$. Alpha-Stabler computes

\begin{equation}
    \operatorname{PSI}_{\ell}^{(t)}
    =
    \frac{
        \left\|X_\ell^{(t)}U_{\ell,r}\right\|_F^2
    }{
        \left\|X_\ell^{(t)}\right\|_F^2+\epsilon
    },
    \qquad
    \operatorname{PSI}^{(t)}
    =
    \frac{
        \sum_{\ell\in\mathcal S}
        \left\|X_\ell^{(t)}U_{\ell,r}\right\|_F^2
    }{
        \sum_{\ell\in\mathcal S}
        \left\|X_\ell^{(t)}\right\|_F^2+\epsilon
    }.
    \label{eq:alpha_online_psi}
\end{equation}

This matrix form aggregates projection energies over tokens without allowing
heterogeneous shifts to cancel. Because it is normalized by the total shift
energy, PSI is insensitive to uniform growth in the magnitude of the activation
change and rises only when a larger fraction rotates into the principal
subspace. In KL-regularized RL, Alpha-Stabler reuses the reference-model forward
pass already required for reference log probabilities; its additional work is
limited to retaining monitored activations and performing the low-rank
projections in Equation~\ref{eq:alpha_online_psi}.

### Predictor: detecting departure from stable training

The Predictor determines when the current update geometry has departed from the
range observed during healthy training. During a short warm-up window, it
estimates a layer-specific baseline using the median $m_\ell$ and robust scale
$s_\ell$:

\begin{equation}
    m_\ell
    =
    \operatorname{median}_{t\leq T_{\mathrm{warm}}}
    \operatorname{PSI}_{\ell}^{(t)},
    \qquad
    s_\ell
    =
    1.4826\,
    \operatorname{median}_{t\leq T_{\mathrm{warm}}}
    \left|
        \operatorname{PSI}_{\ell}^{(t)}-m_\ell
    \right|
    +\epsilon.
    \label{eq:psi_calibration}
\end{equation}

We smooth the online signal using an exponential moving average
$\bar\psi_\ell^{(t)}$ and define two thresholds:

\begin{equation}
    \tau_\ell^{\mathrm{on}}
    =m_\ell+c_{\mathrm{on}}s_\ell,
    \qquad
    \tau_\ell^{\mathrm{safe}}
    =m_\ell+c_{\mathrm{safe}}s_\ell,
    \qquad
    c_{\mathrm{safe}}<c_{\mathrm{on}}.
    \label{eq:psi_thresholds}
\end{equation}

The Predictor raises a warning when any monitored layer remains above
$\tau_\ell^{\mathrm{on}}$ for $p$ consecutive checks. The warm-up calibration
accounts for model- and layer-specific PSI scales, while smoothing and
persistence prevent isolated minibatch fluctuations from being mistaken for
collapse. The lower safe threshold introduces hysteresis: the Controller is
not released immediately after PSI falls marginally below the warning level.

### Controller: restoring the off-principal regime

The Controller is activated only after the Predictor detects sustained
intrusion. Its purpose is not to remove the base model's principal
representation, but to attenuate the principal component of the *RL-induced
change*. For the token-level shift matrix $X_\ell^{(t)}$, this intrusive
component is

\begin{equation}
    X_{\ell,\mathrm{P}}^{(t)}
    =
    \left(X_\ell^{(t)}U_{\ell,r}\right)U_{\ell,r}^{\top},
    \label{eq:principal_shift}
\end{equation}

which can be computed without materializing the full projector. Let
$\psi_\ell^{(t)}=\operatorname{PSI}_\ell^{(t)}$. We retain the following
fraction of the principal component:

\begin{equation}
    \gamma_\ell^{(t)}
    =
    \min\left\{
        1,
        \sqrt{
            \frac{
                \tau_\ell^{\mathrm{safe}}
                \left(1-\psi_\ell^{(t)}\right)
            }{
                \left(1-\tau_\ell^{\mathrm{safe}}\right)
                \psi_\ell^{(t)}+\epsilon
            }
        }
    \right\}.
    \label{eq:controller_gain}
\end{equation}

The controlled activation shift and hidden states are then

\begin{equation}
    \widetilde X_\ell^{(t)}
    =
    X_\ell^{(t)}
    -
    \left(1-\gamma_\ell^{(t)}\right)
    X_{\ell,\mathrm{P}}^{(t)},
    \qquad
    \widetilde H_\ell^{(t)}
    =
    H_\ell^{0,(t)}+\widetilde X_\ell^{(t)}.
    \label{eq:controller_update}
\end{equation}

When $\psi_\ell^{(t)}\leq\tau_\ell^{\mathrm{safe}}$,
$\gamma_\ell^{(t)}=1$ and the Controller is inactive. Otherwise,
Equation~\ref{eq:controller_gain} performs the minimum attenuation required to
return the layer-wise intrusion ratio to the safe level. The same scalar gain
is used within a layer, but each row of $X_{\ell,\mathrm{P}}^{(t)}$ is
_token-specific_. Consequently, the Controller preserves heterogeneous RL
updates rather than replacing them with a common correction. Moreover,

\begin{equation}
    \widetilde X_\ell^{(t)}
    \left(I-U_{\ell,r}U_{\ell,r}^{\top}\right)
    =
    X_\ell^{(t)}
    \left(I-U_{\ell,r}U_{\ell,r}^{\top}\right),
    \label{eq:complement_preservation}
\end{equation}

so the off-principal component associated with effective learning is preserved
exactly. Alpha-Stabler therefore acts only when the update leaves its stable
geometric regime and modifies only the component implicated by the preceding
analysis.

```latex
\begin{algorithm}[t]
\caption{Alpha-Stabler}
\label{alg:alpha_stabler}
\begin{algorithmic}[1]
\Require Policy $f_\theta$, frozen base $f_{\theta_0}$, and PC bases
$\{U_{\ell,r}\}_{\ell\in\mathcal S}$.
\State Calibrate $m_\ell$, $s_\ell$, $\tau_\ell^{\mathrm{on}}$, and
$\tau_\ell^{\mathrm{safe}}$ during warm-up.
\For{training step $t=1,2,\ldots$}
    \State Run the base and current policies on the same token sequences.
    \State Form $X_\ell^{(t)}$ and compute $\operatorname{PSI}_\ell^{(t)}$.
    \State Update the EMA $\bar\psi_\ell^{(t)}$.
    \If{any layer exceeds $\tau_\ell^{\mathrm{on}}$ for $p$ checks}
        \State Raise a warning and activate the Controller.
    \EndIf
    \If{Controller is active}
        \State Compute $X_{\ell,\mathrm{P}}^{(t)}$ and apply
        Equation~\ref{eq:controller_update}.
    \EndIf
    \State Update $\theta$ with the original RL objective and optimizer.
\EndFor
\end{algorithmic}
\end{algorithm}
```

## Evaluation

We evaluate Alpha-Stabler on collapse-prone full-parameter and LoRA RL runs
across two model scales and two task domains. For prediction, we compare PSI
with policy entropy, KL divergence, gradient norm, and score slope, reporting
AUROC, false-alarm rate, and warning lead time. For stabilization, we compare
the complete method with no intervention, gradient clipping, adaptive KL, and
a random-subspace controller. The primary metrics are collapse rate, final
score, score AUC, peak PSI, and wall-clock overhead.

Across \([{\rm TO\ FILL}]\) real runs, PSI provides a median warning lead time
of \([{\rm TO\ FILL}]\) steps and an AUROC of \([{\rm TO\ FILL}]\). Activating
the Controller reduces the collapse rate from \([{\rm TO\ FILL}]\) to
\([{\rm TO\ FILL}]\), while preserving the best score attained before
intervention. The geometric control is more effective than random-subspace
clipping at matched intervention strength, indicating that the benefit is
specific to principal-subspace intrusion rather than generic reduction of the
update norm.

| Method | Collapse rate ↓ | Final score ↑ | Peak PSI ↓ | Overhead ↓ |
|---|---:|---:|---:|---:|
| No intervention | [TO FILL] | [TO FILL] | [TO FILL] | -- |
| Gradient clipping | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Adaptive KL | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Random-subspace control | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| **Alpha-Stabler** | **[TO FILL]** | **[TO FILL]** | **[TO FILL]** | [TO FILL] |

## Ablations

We conduct four compact ablations. **Subspace size** varies the retained PC
fraction \(q\in\{5\%,10\%,20\%,30\%\}\). **Threshold sensitivity** varies
\(c_{\mathrm{on}}\) to quantify the trade-off between false alarms and delayed
intervention. **Geometric specificity** replaces the top-PC subspace with a
random or bottom-PC subspace of the same dimension. **Controller strength**
compares partial attenuation with exact PSI capping. These experiments test
whether stabilization depends on the proposed geometry rather than on generic
clipping or reduced update magnitude.

Alpha-Stabler converts the static off-principal geometry identified by vector
steering into an online training signal and targeted intervention. Although PSI
should be interpreted as an empirical indicator rather than a proven unique
cause of collapse, it provides a measurable bridge from geometric analysis to
practical RL stabilization.

## Recommended two-page layout

- **Top half of page 1:** motivation, Equations (8)--(9), and the two PSI
  collapse panels.
- **Bottom half of page 1:** Predictor, Controller, and the compact algorithm.
- **Top half of page 2:** stabilization curves with and without Alpha-Stabler.
- **Bottom half of page 2:** the compact results table and four ablation
  summaries.
