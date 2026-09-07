# Experimental Setup (LaTeX source)

> 直接粘到 `\subsection{Preliminaries}` 之后即可。所有数值均取自
> `verl/examples/representation/` 下的脚本默认值（`${VAR:-default}`），脚本名在正文中标注。
> 需要的宏包：`booktabs`, `tabularx`, `tcolorbox`（`breakable` 库）, `bbm`（仅
> `\mathbbm{1}` 用到，可换成 `\mathbf{1}`）。

---

```latex
\newpage
\subsection{Experimental Setup}
\label{Experimental Setup}

% ---------------------------------------------------------------------------
\paragraph{Overview.}
Every experiment in this paper instantiates one cell of a
$\text{paradigm}\times\text{update configuration}$ grid. The three
\emph{paradigms} are the ones formalized in \S\ref{Preliminaries}: on-policy RL with
verifiable rewards (Eq.~\eqref{eq:rl_objective}), on-policy distillation
(Eq.~\eqref{eq:opd_objective}), and off-policy distillation
(Eq.~\eqref{eq:offpd_objective}). The three \emph{update configurations} are full
fine-tuning, LoRA, and vector steering, and they differ only in which parameters
receive gradients---the paradigm, data, teacher, sampling parameters, and evaluation
protocol are held fixed within a column so that any difference in outcome is
attributable to the update's dimensionality alone. Concretely, each cell corresponds to one
wrapper script in \texttt{examples/representation/}, all of which delegate to a shared
driver:
\begin{center}\small
\begin{tabular}{@{}llll@{}}
\toprule
& \textbf{Full fine-tuning} & \textbf{LoRA} & \textbf{Vector steering} \\
\midrule
RL      & \texttt{4b\_rl\_fullparam.sh}             & \texttt{4b\_rl\_lora8.sh}             & \texttt{4b\_rl\_single.sh} \\
OPD     & \texttt{4bopd\_fullparam.sh}              & \texttt{4bopd\_lora8.sh}              & \texttt{4bopd\_single.sh} \\
Off-PD  & \texttt{4b\_distill\_offline\_fullparam.sh}& \texttt{4b\_distill\_offline\_lora8.sh}& \texttt{4b\_distill\_offline\_single.sh} \\
\midrule
driver  & \texttt{4b\_rl.sh} & \texttt{4bopd.sh} & \texttt{4b\_distill\_offline.sh} \\
\bottomrule
\end{tabular}
\end{center}
\noindent
The same three-way structure is replicated for the other domains with the prefixes
\texttt{4b\_rl\_science}, \texttt{code\_\{rl,opd,distill\_offline\}},
\texttt{ifevalg\_\{rl,opd,distill\_offline\}}, and
\texttt{deepseek1p5b\_\{rl,opd,distill\_offline\}}.

% ---------------------------------------------------------------------------
\paragraph{Models.}
To ensure the generality of our findings, we conduct experiments across model scales from
1.5B to 14B parameters, spanning two model families (Qwen2.5/DeepSeek-R1-Distill and
Qwen3) and three RL algorithms (PPO, GRPO, DAPO). Table~\ref{mode_config} lists every
base/RL model pair we use. Our experimental models include publicly available
post-RL checkpoints as well as models we train locally with the \texttt{verl}
framework~\citep{Sheng_2025}. For all distillation students in
Table~\ref{mode_config}, the capability-aligned teacher is by default the RL-tuned
version of the student's \emph{own} base model (the RL model listed in the same row), so
that the teacher--student gap is exactly the RL update and nothing else; for
Qwen3-8B-Base we additionally use the larger Qwen3-14B-Base-DAPO as the teacher to check
that our conclusions do not depend on this alignment.

\begin{table}[h]
\centering
\caption{Summary of models considered in this study. ``Open-Source'' indicates whether the
RL checkpoint is publicly released; the remaining checkpoints are trained locally with
\texttt{verl} under the configuration of Table~\ref{tab:hparams}.}
\setlength{\tabcolsep}{4pt}
\label{mode_config}
\begin{tabular}{cccc}
\toprule
\textbf{Base Model} & \textbf{RL Model} & \textbf{Algorithm} & \textbf{Open-Source} \\
\midrule
Qwen2.5-1.5B-Deepseek & JustRL \citep{he2025justrlscaling15bllm} & GRPO & Yes \\
Qwen2.5-1.5B-Deepseek & BroRL \citep{Hu2025BroRLSR} & PPO & Yes \\
Qwen2.5-1.5B-Deepseek & ProRL \citep{Liu2025ProRLPR} & DAPO & Yes \\
Qwen3-4B-Non-Thinking & Qwen3-4B-Non-Thinking-GRPO & GRPO & Yes \\
Qwen2.5-7B & Open-Reasoner-Zero \citep{hu2025openreasonerzeroopensourceapproach} & PPO & Yes \\
Qwen3-8B-Base & Qwen3-8B-PPO \citep{cai2025predictability} & PPO & Yes \\
Qwen3-8B-Base & Qwen3-8B-DAPO \citep{cai2025predictability} & DAPO & Yes \\
Qwen3-14B-Base & Qwen3-14B-Base-DAPO & DAPO & No \\
Qwen3-14B & Qwen3-14B-GRPO & GRPO & No \\
\bottomrule
\end{tabular}
\end{table}

% ---------------------------------------------------------------------------
\paragraph{Datasets and evaluation.}
The training data spans four topics, each with its own verifier and held-out evaluation
set (Table~\ref{tab:data}). Rewards are always \emph{verifiable} in the sense of
\S\ref{Preliminaries}: rule-based answer matching for mathematical and scientific
reasoning, unit-test execution for code, and programmatic constraint checking for
instruction following. No learned reward model is used anywhere in this paper. Following
DAPO, mathematical training prompts are filtered to the hardest difficulty bucket
(\texttt{level6}) so that the reward signal is not saturated at initialization.

\begin{table}[h]
\centering\small
\caption{Training and evaluation data per domain, with the reward manager used by
\texttt{verl}. \texttt{naive} performs rule-based answer verification; \texttt{dapo}
additionally applies overlong reward shaping and sandboxed unit-test execution.}
\label{tab:data}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lllll@{}}
\toprule
\textbf{Domain} & \textbf{Train set} & \textbf{Evaluation} & \textbf{Reward} & \textbf{Manager} \\
\midrule
Math & DeepMath-103K \citep{he2025deepmath103klargescalechallengingdecontaminated} & AIME 2024, AIME 2025 & answer match & \texttt{naive} \\
     & (\texttt{level6} subset), MATH-12K \citep{lightman2023lets} & & & \\
Science & OpenScienceReasoning-2 \citep{} (\texttt{level6}), & GPQA \citep{}, & answer match & \texttt{dapo} \\
        & SciKnowEval \citep{feng2024sciknoweval} & held-out split & & \\
Code & Eurus \citep{cui2025process} & LiveCodeBench-v5 \citep{}, & unit tests & \texttt{dapo} \\
     & & held-out split & & \\
Instr.\ following & IFEval-G \citep{}, derived from the & held-out split & constraint check & \texttt{naive} \\
        & instruction-following subset of Nemotron & & & \\
        & RL-Training-Blend \citep{} & & & \\
\bottomrule
\end{tabular}
\end{table}

% ---------------------------------------------------------------------------
\paragraph{Update configurations.}
The three configurations are the only thing that varies within a paradigm.

\begin{itemize}[leftmargin=1.4em,itemsep=2pt]
\item \textbf{Full fine-tuning} updates all $|\theta|$ parameters. This is the reference
point: whatever the RL update is, full fine-tuning can in principle express it.

\item \textbf{LoRA} \citep{hu2021loralowrankadaptationlarge} injects rank-$r$ adapters
$\Delta W = BA$ with $r=8$, $\alpha=16$, on \texttt{all-linear} target modules, leaving
the backbone frozen. This restricts the update to a low-rank---but still
\emph{context-dependent}---function of the input.

\item \textbf{Vector steering} freezes \emph{all} weights and trains only a set of
context-invariant bias vectors added to the residual stream. For a controlled layer set
$\mathcal S$, the forward pass becomes
\begin{equation}
\bm h_\ell \;\mapsto\; \bm h_\ell + \sum_{k=1}^{K} c_k\, \bm v^{(k)}_\ell,
\qquad \bm v^{(k)}_\ell \in \mathbb{R}^{d},\quad \ell \in \mathcal S ,
\label{eq:vector_steering}
\end{equation}
so the entire trainable parameter count is $K\,|\mathcal S|\,d$---for the single-vector
case $K=1$ on Qwen3-4B ($d=2560$) with $|\mathcal S|=10$ this is $2.6\times10^4$ scalars,
roughly $6\times10^{-6}$ of the model. Crucially, $\bm v^{(k)}_\ell$ carries no dependence
on the input: the \emph{same} vector is added at every context and every decoding
position, which is precisely the constraint whose consequences \S\ref{sec:setup} analyzes.
\end{itemize}

\noindent
Unless stated otherwise the controlled layer set is $\mathcal S=\{5,\dots,14\}$, i.e.\ the
ten decoder layers following the first quarter of the network; \S\ref{sec:layers} reports
the sweep over $\mathcal S$ that motivates this choice. Our default is the single-vector
case ($K=1$, \texttt{mode=single},
\texttt{learnable\_alpha=false}, \texttt{alpha\_init=0}), i.e.\ one raw vector trained
from zero with no separate gain scalar. Multi-vector runs ($K>1$) initialize the basis by
sampling from the positive first orthant of the unit hypersphere
(\texttt{sampling\_method=hypersphere}) at scale $0.1$, and use the
\texttt{sequential\_orthogonal} curriculum: $\bm v^{(0)}$ is trained alone for
\texttt{SEQ\_RAW\_STEPS}$=100$ steps and frozen, after which each subsequent
$\bm v^{(k)}$ is trained with its gradient projected onto the orthogonal complement of
$\operatorname{span}\{\bm v^{(j)}\}_{j<k}$, switching to $\bm v^{(k+1)}$ when the loss
plateaus (patience $10$, per-vector cap \texttt{SEQ\_MAX\_ITERS}$=500$). The steering
hook is applied to response tokens only during RL
(\texttt{force\_all\_tokens=false}) and to all tokens during distillation
(\texttt{force\_all\_tokens=true}), since in the latter case the prefix is also
supervised. Because vector steering trains $O(d)$ rather than $O(|\theta|)$ parameters,
it requires a much larger learning rate than the other two configurations; the values we
use are given in Table~\ref{tab:hparams}.

% ---------------------------------------------------------------------------
\paragraph{RL training.}
For models we train locally we use \texttt{verl}~\citep{Sheng_2025} and follow the
reference setups of the corresponding papers. All RL runs share a core configuration:
maximum prompt length $2{,}048$ and maximum response length $20{,}480$ tokens (total
budget $22{,}528$; $\texttt{max\_position\_embeddings}=32{,}768$), a rollout of $n=16$
samples per prompt, a training prompt batch of $128$ prompts, and a mini-batch of $32$ for
each optimizer step, under \texttt{bfloat16} with gradient clipping $1.0$ and sequence
parallelism of degree $8$. Sampling during training is unbiased
($\text{temperature}=1.0$, $\text{top-}p=1.0$, $\text{top-}k$ disabled); validation
decodes with $\text{top-}p=0.7$. We monitor the mean reward per training batch and
terminate once it fails to improve for five consecutive evaluations.

Each algorithm then adds its own hyperparameters. For \textbf{GRPO} we set both clipping
ratios to $0.2$ and apply a KL loss with coefficient $0.001$, following
\citet{deepseekai2025deepseekr1incentivizingreasoningcapability}. For \textbf{DAPO} we
enable clip-higher, dynamic sampling, token-level policy-gradient loss, and overlong
reward shaping, with the recommended hyperparameters of
\citet{yu2025dapoopensourcellmreinforcement}: $\epsilon_{\text{low}}=0.2$,
$\epsilon_{\text{high}}=0.28$, an overlong buffer of $2{,}048$ tokens with penalty
factor $1.0$, dynamic sampling that resamples groups with zero accuracy variance (up to
$10$ generation batches per step), and the KL term removed entirely
($\beta=0$ in Eq.~\eqref{eq:rl_objective}). \textbf{PPO} results are taken from the
released checkpoints in Table~\ref{mode_config} rather than retrained.

% ---------------------------------------------------------------------------
\paragraph{On-policy distillation.}
We follow the setting of \citet{Yang2026LearningBT}. The student generates its own
rollouts ($n=1$ per prompt, $\text{temperature}=1.0$, $\text{top-}p=1.0$) with maximum
prompt length $3{,}072$ and response length $16{,}384$; the teacher is loaded as the
reference model and scored on those same student trajectories, so that the per-token
advantage is the negated reverse KL
$A_t = \log \pi^{*}(y_t\mid x,y_{<t}) - \log \pi_\theta(y_t\mid x,y_{<t})$, which is exactly
the token-level gradient of Eq.~\eqref{eq:opd_gradient_approx}. This is implemented as
\texttt{policy\_loss.only\_reverse\_kl\_advantages=true} on top of the GRPO advantage
estimator; the separate KL-penalty term is disabled
(\texttt{kl\_loss\_coef}$=0$, \texttt{use\_kl\_in\_reward=false}) and the entropy bonus is
zero, so reverse KL to the teacher is the \emph{only} learning signal. Because rollouts
are generated by vLLM while log-probabilities are recomputed by the FSDP worker, we
correct the resulting numerical mismatch with token-level importance sampling truncated
at a ratio of $5.0$. Prompt batch size is $1{,}024$ with a single optimizer step per
batch (\texttt{ppo\_mini\_batch\_size}$=1{,}024$), for $10$ epochs.

% ---------------------------------------------------------------------------
\paragraph{Off-policy distillation.}
The off-policy runs use the identical loss and trainer, changing only \emph{who samples
the trajectories}, so the comparison isolates the state-distribution mismatch of
Eq.~\eqref{eq:is_ratio}. We first dump teacher rollouts once
(\texttt{gen\_teacher\_rollouts.sh}): $2$ full sampling passes over the prompt set at
$\text{temperature}=1.0$, $\text{top-}p=1.0$, one sample per prompt per pass, generation
batch $1{,}024$, with the same $3{,}072/16{,}384$ length budget. The per-round dumps are
exploded into one $(\text{prompt},\text{response})$ pair per generation by
\texttt{merge\_rollouts\_to\_sft.py} and merged into a single corpus. During training,
vLLM sampling is replaced by an offline loader that reads and tokenizes this corpus as
the rollout batch, after which the standard distillation loop runs unchanged (student
forward $\to$ teacher forward $\to$ reverse-KL advantage $\to$ policy update). Since the
behaviour distribution is fixed, rollout importance correction is disabled
(\texttt{rollout\_is=null}). Everything else matches the on-policy configuration; we
train for more epochs ($100$ vs.\ $10$) because each epoch costs no generation.

% ---------------------------------------------------------------------------
\begin{table}[h]
\centering\small
\caption{Learning rates and paradigm-specific hyperparameters. Within a paradigm only the
learning rate changes across update configurations; vector steering requires a learning
rate $3$--$4$ orders of magnitude larger than full fine-tuning because it optimizes
$O(d)$ instead of $O(|\theta|)$ parameters.}
\label{tab:hparams}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lccc@{}}
\toprule
& \textbf{RL} & \textbf{OPD} & \textbf{Off-PD} \\
\midrule
LR, full fine-tuning & $1\times10^{-6}$ & $1\times10^{-5}$ & $5\times10^{-5}$ \\
LR, LoRA ($r{=}8$)   & $3\times10^{-5}$ & $5\times10^{-6}$ & $1\times10^{-5}$ \\
LR, vector steering  & $5\times10^{-2}$ & $1\times10^{-1}$ & $5\times10^{-2}$ \\
\midrule
LR warmup & yes & none & none \\
Max prompt / response & $2{,}048$ / $20{,}480$ & $3{,}072$ / $16{,}384$ & $3{,}072$ / $16{,}384$ \\
Prompt batch & $128$ & $1{,}024$ & $1{,}024$ \\
Mini-batch (per step) & $32$ & $1{,}024$ & $1{,}024$ \\
Samples per prompt $n$ & $16$ & $1$ & $1$ (pre-generated) \\
Rollout source & student & student & frozen teacher \\
Rollout IS correction & --- & token, threshold $5.0$ & off \\
KL loss coefficient & $0.001$ (GRPO) / $0$ (DAPO) & $0$ & $0$ \\
Epochs & $10$ & $10$ & $100$ \\
Precision & \texttt{bfloat16} & \texttt{bfloat16} & \texttt{bfloat16} \\
\bottomrule
\end{tabular}
\end{table}

% ---------------------------------------------------------------------------
\paragraph{Prompts.}
All Qwen3 models are run and trained in \emph{non-thinking} mode
(\texttt{enable\_thinking=False} passed to the chat template) so that the response
distribution is directly comparable between the base model, the RL teacher, and every
steered student; DeepSeek-R1-Distill students keep thinking mode enabled, as their base
model is trained that way. We always use each model's own built-in chat template rather
than a hand-written wrapper, so no additional system prompt is introduced. Each domain
appends its own answer-format instruction, which is also what the corresponding verifier
parses; the four templates are given below, with \texttt{\{question\}} the raw problem
statement and \texttt{\{CoT\}} the sampled response.

For \textbf{mathematical reasoning} the answer-extraction instruction is appended to the
question:
\begin{verbatim}
User:
{question}
Please reason step by step, and put your final answer within \boxed{}.
<think>
</think>
Assistant: {CoT}
\end{verbatim}

\noindent
For \textbf{scientific reasoning} the instruction is placed first and the question is a
multiple-choice item whose options are enumerated A--J, so that the verifier can match a
single letter:
\begin{verbatim}
User:
Solve the following problem. Make sure to put the answer (and only answer)
inside \boxed{}.

{question}

A: {option_A}
B: {option_B}
...
J: {option_J}
<think>
</think>
Assistant: {CoT}
\end{verbatim}

\noindent
For \textbf{code generation} the instruction requires a single fenced Python block, which
the sandbox extracts verbatim and executes against the hidden unit tests:
\begin{verbatim}
User:
{question}

Write Python code to solve the problem. Present the code in
```python
Your code
```
at the end.
You need to think first then write the Python code.
<think>
</think>
Assistant: {CoT}
\end{verbatim}

\noindent
For \textbf{instruction following} the constraints are stated in natural language at the
head of the prompt and no reasoning instruction is added, since the verifiable
constraints refer to the surface form of the response and a reasoning preamble would
itself violate them:
\begin{verbatim}
User:
Please follow these instructions in your response: {constraint_1} and
{constraint_2} and ... {constraint_n}. {question}
<think>
</think>
Assistant: {CoT}
\end{verbatim}

\noindent
Teacher rollouts for off-policy distillation are generated with the \emph{same} template
as the corresponding on-policy runs, so the two paradigms differ only in the sampling
policy.

% ---------------------------------------------------------------------------
\paragraph{Infrastructure.}
All training runs use $8\times$ or $32\times$ H20 96GB GPUs with FSDP for the actor and
vLLM for generation. Vector-steering runs disable CUDA graph capture
(\texttt{enforce\_eager=true}) because the steering hook mutates hidden states inside the
vLLM forward pass; LoRA and full-parameter runs leave CUDA graphs enabled. Rollout tensor
parallelism is $4$ for RL and $2$ for distillation, and the reference (teacher) model is
kept parameter-offloaded to CPU between forward passes. The three commands below give the
exact invocation for each paradigm in the vector-steering configuration; the
full-parameter and LoRA variants differ only in the overrides listed in
Table~\ref{tab:hparams}.
```

---

## RL 命令（vector steering）

```latex
\begin{tcolorbox}[
    colback=gray!5, colframe=gray!70, arc=3pt,
    left=2pt, right=2pt, top=2pt, bottom=2pt, boxrule=0.5pt,
    breakable, fontupper=\small, listing only,
    listing options={language=bash},
    title=RL Training Command (vector steering), title style={color=black},
    label=cmd:rl
]
\begin{verbatim}
# examples/representation/4b_rl_single.sh -> 4b_rl.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric=acc \
    +algorithm.filter_groups.max_num_gen_batches=10 \
    data.train_files=/path/to/DeepMath-103K/train_filtered_level6.parquet \
    data.val_files="['/path/to/AIME2024/test.parquet','/path/to/AIME2025/test.parquet']" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.train_batch_size=128 \
    +data.gen_batch_size=128 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_trainable_token_vector=true \
    actor_rollout_ref.model.trainable_token_vector_mode=single \
    actor_rollout_ref.model.trainable_token_vector_num=1 \
    actor_rollout_ref.model.trainable_token_vector_layer_start=5 \
    actor_rollout_ref.model.trainable_token_vector_layer_end=15 \
    actor_rollout_ref.model.trainable_token_vector_sampling_method=hypersphere \
    actor_rollout_ref.model.trainable_token_vector_scale=0.1 \
    actor_rollout_ref.model.trainable_token_vector_learnable_alpha=false \
    actor_rollout_ref.model.trainable_token_vector_alpha_init=0.0 \
    actor_rollout_ref.model.trainable_token_vector_curriculum=none \
    actor_rollout_ref.model.trainable_token_vector_force_all_tokens=false \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    reward_model.reward_manager=naive \
    +reward_model.overlong_buffer.enable=True \
    +reward_model.overlong_buffer.len=2048 \
    +reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.save_vector=true \
    trainer.save_vector_dir=/path/to/trainable_vectors \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.test_freq=5 \
    trainer.save_freq=25 \
    trainer.total_epochs=10 $@
\end{verbatim}
\end{tcolorbox}
```

---

## OPD 命令（vector steering）

```latex
\begin{tcolorbox}[
    colback=gray!5, colframe=gray!70, arc=3pt,
    left=2pt, right=2pt, top=2pt, bottom=2pt, boxrule=0.5pt,
    breakable, fontupper=\small, listing only,
    listing options={language=bash},
    title=On-Policy Distillation Command (vector steering), title style={color=black},
    label=cmd:opd
]
\begin{verbatim}
# examples/representation/4bopd_single.sh -> 4bopd.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    data.train_files=/path/to/DeepMath-103K/train_filtered_level6.parquet \
    data.val_files="['/path/to/AIME2024/test.parquet','/path/to/AIME2025/test.parquet']" \
    data.train_batch_size=1024 \
    data.max_prompt_length=3072 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.base_model_path=$MODEL_PATH \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_trainable_token_vector=true \
    actor_rollout_ref.model.trainable_token_vector_mode=single \
    actor_rollout_ref.model.trainable_token_vector_num=1 \
    actor_rollout_ref.model.trainable_token_vector_layer_start=5 \
    actor_rollout_ref.model.trainable_token_vector_layer_end=15 \
    actor_rollout_ref.model.trainable_token_vector_sampling_method=hypersphere \
    actor_rollout_ref.model.trainable_token_vector_scale=0.1 \
    actor_rollout_ref.model.trainable_token_vector_learnable_alpha=false \
    actor_rollout_ref.model.trainable_token_vector_alpha_init=0.0 \
    actor_rollout_ref.model.trainable_token_vector_curriculum=none \
    actor_rollout_ref.model.trainable_token_vector_force_all_tokens=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    trainer.save_vector=true \
    trainer.save_vector_dir=/path/to/trainable_vectors \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 $@
\end{verbatim}
\end{tcolorbox}
```

---

## Off-policy distillation 命令（vector steering，含 teacher rollout 生成）

```latex
\begin{tcolorbox}[
    colback=gray!5, colframe=gray!70, arc=3pt,
    left=2pt, right=2pt, top=2pt, bottom=2pt, boxrule=0.5pt,
    breakable, fontupper=\small, listing only,
    listing options={language=bash},
    title=Off-Policy Distillation Command (vector steering), title style={color=black},
    label=cmd:offpd
]
\begin{verbatim}
# Stage 1: dump teacher rollouts once (examples/representation/gen_teacher_rollouts_4b.sh)
python3 -m verl.trainer.main_generation \
    model.path=$TEACHER_MODEL_PATH \
    data.path=/path/to/DeepMath-103K/train_filtered_level6.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=1024 \
    rollout.temperature=1.0 \
    rollout.top_p=1.0 \
    rollout.prompt_length=3072 \
    rollout.response_length=16384 \
    rollout.max_num_batched_tokens=32768 \
    rollout.tensor_model_parallel_size=2 \
    data.output_path=$OUTPUT_BASE/round_${i}.parquet     # repeated for i = 0, 1

python3 examples/representation/merge_rollouts_to_sft.py \
    --inputs $OUTPUT_BASE/round_*.parquet \
    --output $OUTPUT_BASE/teacher_sft_all.parquet \
    --prompt_key prompt --response_key response

# Stage 2: distill off-policy on the frozen corpus
#   (examples/representation/4b_distill_offline_single.sh -> 4b_distill_offline.sh)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.use_kl_in_reward=False \
    data.train_files=$OUTPUT_BASE/teacher_sft_all.parquet \
    data.val_files="['/path/to/AIME2024/test.parquet','/path/to/AIME2025/test.parquet']" \
    data.train_batch_size=1024 \
    data.max_prompt_length=3072 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_trainable_token_vector=true \
    actor_rollout_ref.model.trainable_token_vector_mode=single \
    actor_rollout_ref.model.trainable_token_vector_num=1 \
    actor_rollout_ref.model.trainable_token_vector_layer_start=5 \
    actor_rollout_ref.model.trainable_token_vector_layer_end=15 \
    actor_rollout_ref.model.trainable_token_vector_sampling_method=hypersphere \
    actor_rollout_ref.model.trainable_token_vector_scale=0.1 \
    actor_rollout_ref.model.trainable_token_vector_learnable_alpha=false \
    actor_rollout_ref.model.trainable_token_vector_curriculum=none \
    actor_rollout_ref.model.trainable_token_vector_force_all_tokens=true \
    actor_rollout_ref.actor.optim.lr=5e-2 \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.offline_teacher_rollout=true \
    actor_rollout_ref.rollout.offline_teacher_data_path=$OUTPUT_BASE/teacher_sft_all.parquet \
    actor_rollout_ref.rollout.offline_teacher_response_key=response \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    trainer.save_vector=true \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 $@
\end{verbatim}
\end{tcolorbox}
```

---

## 需要你确认/补齐的点

1. **五个 `\citep{}` 是空的**（OpenScienceReasoning-2、GPQA、LiveCodeBench-v5、IFEval-G、Nemotron RL-Training-Blend），按你的要求留空，编译前把 `.bib` 里的 key 填进去。Table~\ref{tab:data} 的 instruction-following 行我按 `data/ifevalg/convert_ifevalg_train.py` 改成了「derived from the instruction-following subset of Nemotron RL-Training-Blend」（该脚本读的是 NVIDIA `Nemotron-3-Nano-RL-Training-Blend` 的 `nano_v3_sft_profiled_instruction_following` 子集），**不是** WildChat——如果这个数据集在正文里另有正式叫法，替换掉这句。
2. **`\ref{Preliminaries}` / `\eqref{eq:offpd_objective}` / `\eqref{eq:is_ratio}`**：引用的是上一轮我给你的 off-policy 段落里的 label，若你没采用那段需改。
3. **14B 的 RL 超参**：Table~\ref{tab:hparams} 里的数值取自 `4b_rl.sh`（4B）。14B 本地训练的脚本不在 `representation/` 下（在 `recipe/dapo/`），若表里要单列 14B 的配置，我需要再看那个脚本。
4. **Infrastructure 里的 "$8\times$ or $32\times$ H20"**：脚本默认都是 `nnodes=1, n_gpus_per_node=8`，如果所有汇报的实验都是单机 8 卡，把 `32\times` 那半句删掉更稳。

已按你的四条修改：向量 OPD 学习率统一为 $1\times10^{-1}$；注入层三个命令块统一为 `5:15`（正文写成 $\mathcal S=\{5,\dots,14\}$，参数量按 $|\mathcal S|=10$ 重算）；code / instruction-following 的 prompt 也贴成 `verbatim`（四个模板都是从对应 parquet 的真实 `prompt` 字段取的）；不确定的 citation 留空。

已写入 `analysis/vector_study/experimental_setup.md`。

