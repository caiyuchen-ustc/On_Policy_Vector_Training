# On-Policy Vector Training

RL teacher training, on/off-policy distillation, trainable representation vectors,
and Alpha-Stabler on top of [verl](https://github.com/volcengine/verl).
This repository accompanies **Learning to Steer, Steering to See: Unveiling the
Geometry of RLVR in Large Language Models via Trainable Vectors**.

The repository provides training scripts for four task domains, representation
analysis, and principal-subspace monitoring and control.

## Layout

```text
configs/                    public teacher IDs and pinned dataset revisions
scripts/data/               byte-preserving dataset download/upload/verification
scripts/rl_teacher/         RL entry point
scripts/opd/                on-policy distillation entry point
scripts/offpolicy/          teacher generation and offline distillation
scripts/alpha_stabler/      online geometry monitoring and RL control
scripts/train.py            shared configuration and CLI for all four domains
analysis/                   training curves and representation geometry
examples/representation/    experiment variants
recipe/dapo/                DAPO trainer and configuration
recipe/code_sandbox/        code reward execution service
verl/                      training library, steering hooks and Alpha-Stabler
tests/opv/                  focused CPU checks
```

## Installation

Use Python 3.10+ with compatible CUDA, PyTorch, vLLM and FlashAttention versions.
Training uses FSDP and synchronous vLLM. See the supplied `docker/` images and
[installation guide](https://verl.readthedocs.io/en/latest/start/install.html).

```bash
pip install -e .
pip install -r requirements-opv.txt
# Install the matching vLLM and flash-attn builds for your CUDA/PyTorch versions.
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng
```

Model and dataset downloads use Hugging Face authentication (`HF_TOKEN` if needed).
Enable W&B with `--wandb` and set `WANDB_API_KEY` in your environment.

## Data and pretrained teachers

Download models and datasets from the Hugging Face repositories below. GitHub
contains the code and documentation; weights, datasets and checkpoints stay outside
the repository. Launchers use the public model IDs by default.

Parquet files retain their original format and contents. Train/evaluation files have
separate Hub configurations to support their different Arrow schemas.

| Task | Dataset | Training rows | Evaluation | Teacher |
| --- | --- | ---: | --- | --- |
| Math | [OPV-Math](https://huggingface.co/datasets/caiyuchen/OPV-Math) | 14,116 DAPO teacher / 57,046 DeepMath distillation | AIME2024: 30; AIME2025: 30 | [Qwen3-4B Math RL](https://huggingface.co/caiyuchen/Qwen3-4B-Non-Thinking-Math-RL) |
| Science | [OPV-Science](https://huggingface.co/datasets/caiyuchen/OPV-Science) | 3,901 | SciKnowEval: 434 | [DeepSeek-Qwen 1.5B Science RL](https://huggingface.co/caiyuchen/Qwen2.5-1.5B-Dpsk-Science-RL) |
| Code | [OPV-Code](https://huggingface.co/datasets/caiyuchen/OPV-Code) | 25,276 | LiveCodeBench v5: 880; Eurus: 1,024 | [Qwen3-8B Code RL](https://huggingface.co/caiyuchen/Qwen3-8B-Code-RL) |
| Instruction | [OPV-Instruction](https://huggingface.co/datasets/caiyuchen/OPV-Instruction) | 16,575 | IFBench: 300 | [DeepSeek-Qwen 7B IF RL](https://huggingface.co/caiyuchen/Qwen2.5-7B-Dpsk-Instruct-Following-RL) |

```bash
python scripts/data/manage.py download --pins configs/datasets.json
python scripts/data/verify_hub.py
```

Use `--train-file` and `--val-files` for local datasets. Dataset cards and manifests
document the sources, splits and file hashes. `scripts/data/manage.py prepare`
copies original files into a staging directory; `publish` uploads those files.

## 1. RL teacher training

`DOMAIN` is `math`, `science`, `code`, or `instruction`. Default base models are
Qwen3-4B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-8B, and
DeepSeek-R1-Distill-Qwen-7B respectively. Override any model with `--model`.

```bash
bash scripts/rl_teacher/run.sh math --method full --rl-algorithm dapo
bash scripts/rl_teacher/run.sh science --method full
bash scripts/rl_teacher/run.sh instruction --method lora --lora-rank 8
bash scripts/rl_teacher/run.sh code --method full --sandbox-url http://127.0.0.1:8080/run_code
```

The DAPO option uses `recipe.dapo.main_dapo`, including dynamic group filtering.
GRPO uses the existing PPO trainer. Code reward execution follows the existing
[sandbox instructions](recipe/code_sandbox/README.md); run generated code in an
isolated execution environment. Default evaluation uses four samples and all rows;
`--val-max-samples 100` provides the original quick-evaluation option.

Merge a full FSDP checkpoint for inference or upload:

```bash
bash scripts/merge.sh outputs/math-rl-full/checkpoints/global_step_500/actor models/math-teacher
```

The four teachers above are already available; retraining is optional for distillation.

## 2. OPD and representation training

```bash
bash scripts/opd/run.sh math --method full
bash scripts/opd/run.sh math --method lora --lora-rank 8
bash scripts/opd/run.sh math --method vector --layers 5:20 --lr 0.1
bash scripts/opd/run.sh science --method sequential --vectors 8 --stage-steps 100
bash scripts/opd/run.sh instruction --method gated --layers 20:21 --gate-rank 4
```

Layer indices are zero-based and ranges are inclusive. A vector is shared across
tokens at its decoder layer. Sequential mode reuses the existing orthogonal-vector
curriculum, activating one direction at a time; gated mode reuses the existing
input-dependent gate. Full/LoRA/vector checkpoints and per-step vectors are written
under `outputs/<domain>-<stage>-<method>/`.

## 3. Off-policy distillation

Generate the frozen teacher responses once, merge the original response-list format,
then use the existing offline rollout path:

```bash
bash scripts/offpolicy/generate.sh science
python scripts/merge_rollouts.py \
  --inputs outputs/science-generate-full/round_0.parquet \
  --output outputs/science-generate-full/teacher_sft.parquet
bash scripts/offpolicy/run.sh science --method vector --lr 0.05 \
  --offline-data outputs/science-generate-full/teacher_sft.parquet
```

Use `--teacher` for a different teacher. Offline training uses reverse-KL weighting
on teacher-generated trajectories. Validation generates responses with the current student.

## 4. Analysis and Alpha-Stabler

The [analysis scripts](analysis/README.md) cover training curves and representation
geometry. Training logs validation metrics and PSI to `outputs/<run>/metrics.jsonl`.

```bash
bash scripts/alpha_stabler/run.sh
python analysis/plot_metrics.py outputs/science-alpha-full/metrics.jsonl --list-keys
python analysis/plot_metrics.py outputs/science-alpha-full/metrics.jsonl \
  --keys alpha_stabler/psi alpha_stabler/active_layers
```

Alpha-Stabler calibrates frozen principal bases, monitors actor/base activation
shifts and projects backward gradients when intrusion persists. See the
[Alpha-Stabler guide](docs/alpha_stabler.md) for configuration and checkpoint handling.
The default command selects SciKnowEval and downloads missing model/data files.
Use `--check` to validate the training environment and inputs before launching.

## Configuration and checks

All launchers accept `--dry-run` to show the command, `--config-only` to compose the
Hydra configuration without running training, and explicit `key=value` overrides.
Use `--gpus`, `--nodes`, `--tp`, `--batch-size`, `--mini-batch`, `--steps`, `--lr`,
`--name`, `--output-dir`, `--eval-every`, and `--save-every` for routine settings.

```bash
python scripts/train.py opd code --method vector --config-only
python -m pytest -q tests/opv
python scripts/check_repository.py
```

See [testing](docs/validation.md) for test coverage and GPU checks, and
`examples/representation/` for individual experiment settings. The verl foundation
retains its Apache-2.0 license and [upstream documentation](docs/UPSTREAM_README.md).
