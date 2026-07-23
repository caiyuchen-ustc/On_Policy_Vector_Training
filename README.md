# On-Policy Vector Training (G-OPD)

This repository is a [verl](https://github.com/volcengine/verl) fork implementing
**G-OPD** (On-Policy Distillation with trainable token / representation vectors) and
the associated **representation training** scripts.

It contains the full `verl` training stack plus custom reward verifiers and a set of
ready-to-run launch scripts under [`examples/representation/`](examples/representation/)
covering multiple base models, domains, and training modes
(full-parameter / LoRA / vector-only OPD).

> The verl upstream README is preserved at [`README.verl.md`](README.verl.md).

## Repository layout

```
verl/                         # the verl python package (installable)
├── trainer/main_ppo.py       # main entrypoint: python3 -m verl.trainer.main_ppo
├── utils/reward_score/       # custom reward verifiers (see below)
└── ...
examples/
├── representation/           # >>> the G-OPD launch scripts (main entrypoints) <<<
├── g_opd/                    # G-OPD helpers / configs
└── eff_opd/                  # efficient-OPD helpers + hard_100.parquet test fixture
recipe/                       # verl recipes
scripts/                      # utility scripts (model merge, config gen, ...)
```

## Installation

```bash
# create env (Python 3.10), then install verl in editable mode
pip install -e .
# plus the training/inference stack expected by the scripts:
#   torch 2.6, vllm, flash-attn 2.7.4, flashinfer 0.2.2, ray, wandb
# see requirements*.txt for the full dependency lists
```

The scripts were run with a conda env (`verlopd`) and `vllm` as the rollout backend
(`VLLM_ATTENTION_BACKEND=XFORMERS`).

## The representation scripts

All launch scripts live in [`examples/representation/`](examples/representation/).
Each domain/model has up to four variants:

| Suffix          | Training mode                                                        |
|-----------------|----------------------------------------------------------------------|
| `*_opd*.sh`     | On-Policy Distillation with a teacher model                          |
| `*_fullparam.sh`| Full-parameter training (`LORA_RANK=0`, no vector)                   |
| `*_lora8.sh`    | LoRA training (`LORA_RANK=8`)                                        |
| `*_single*.sh`  | Single-GPU / single-node convenience wrapper                         |

Vector-only training is selected by `ENABLE_TRAINABLE_TOKEN_VECTOR=true` with
`LORA_RANK=0`.

### Models / domains covered

- **Math**: `4b*` (Qwen3-4B), `1p5b*` / `deepseek1p5b*` (DeepSeek-R1-Distill-Qwen-1.5B)
- **Code**: `code_*`
- **Science**: `4b_rl_science*`, `sciknoweval` (via `sciknoweval_verify.py`)
- **Instruction following**: `ifevalg_*` (via `ifevalg_verify.py`)
- **Logic / KK puzzles**: `logic_*` (via `kk_verify.py`)
- **Verifiable reasoning**: `verif_*` (via `verif_verify.py`)
- **Agent / tool use**: `agent_rl*` (uses `workbench_tool_config.yaml`)

### Custom reward verifiers

Scripts wire rewards through `custom_reward_function.path`, pointing at:

- `verl/utils/reward_score/ifevalg_verify.py`
- `verl/utils/reward_score/kk_verify.py`
- `verl/utils/reward_score/sciknoweval_verify.py`
- `verl/utils/reward_score/verif_verify.py`
- `verl/utils/reward_score/synlogic_verify.py`, `math_verify.py`, `scicence_verify.py`

## Running

The wrapper scripts (`*_fullparam.sh`, `*_lora8.sh`, `*_single.sh`) set defaults and
then `exec` the main per-domain script (e.g. `4bopd.sh`). Every parameter is an
overridable environment variable. Example:

```bash
cd examples/representation
# full-parameter OPD on Qwen3-4B (math)
bash 4bopd_fullparam.sh

# LoRA-8 OPD
bash 4bopd_lora8.sh

# override any knob inline
ACTOR_LR=1e-6 EXPERIMENT_NAME=my_run bash 4bopd_fullparam.sh
```

## ⚠️ Paths and data (must configure)

The scripts contain **absolute paths** from the original training cluster that you
**must adjust** for your environment:

- **conda env**: `source .../anaconda3/bin/activate .../envs/verlopd`
- **base / teacher models**: `MODEL_PATH`, `Teacher_MODEL_PATH`
- **training / eval data** (`.parquet`): under `.../G-OPD/data/G-OPD-Training-Data/...`
  (DeepMath-103K, AIME2024/2025, Eurus code, open_science_reasoning, sciknoweval, ...)
- **output / checkpoint dirs**: `LOCAL_DIR_BASE`, `SAVE_VECTOR_BASE_DIR`
- **`WANDB_API_KEY`** and proxy env vars

These are **not** shipped in this repo (datasets and checkpoints are large / private).
Point the variables at your own copies before launching.
