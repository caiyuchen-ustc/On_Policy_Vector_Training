# Experiment variants

These scripts contain individual experiment settings. Set `OPV_ROOT`,
`OPV_LEGACY_DATA_ROOT` and `OPV_LEGACY_MODEL_ROOT` to match your local directories.
Use `scripts/train.py` for the shared training interface.

| Original family | Portable command |
| --- | --- |
| `4b_rl*`, `deepseek1p5b_rl*`, `code_rl*`, `ifevalg_rl*` | `python scripts/train.py rl DOMAIN --method full` |
| `4bopd*`, `deepseek1p5b_opd*`, `code_opd*`, `ifevalg_opd*` | `python scripts/train.py opd DOMAIN --method vector` |
| `gen_teacher_rollouts*` | `python scripts/train.py generate DOMAIN` |
| `*_distill_offline*` | `python scripts/train.py offpolicy DOMAIN --offline-data FILE` |
| `*_seqbasis*` | add `--method sequential --vectors 8 --stage-steps 100` |
| `*gated*` | add `--method gated --gate-rank 1` |

`DOMAIN` is `math`, `science`, `code`, or `instruction`. Offline distillation uses
teacher-generated trajectories with a reverse-KL surrogate.
