# Historical experiment launchers

These are the original experiment variants, retained for their settings and analysis
provenance. Embedded credentials, the forced Conda activation, and cluster-specific
network settings have been removed. They use `OPV_ROOT`, `OPV_LEGACY_DATA_ROOT` and
`OPV_LEGACY_MODEL_ROOT` where possible. Some variants still require checkpoint paths
or old folder layouts; use the portable `scripts/train.py` entry point for new runs.

| Original family | Portable command |
| --- | --- |
| `4b_rl*`, `deepseek1p5b_rl*`, `code_rl*`, `ifevalg_rl*` | `python scripts/train.py rl DOMAIN --method full` |
| `4bopd*`, `deepseek1p5b_opd*`, `code_opd*`, `ifevalg_opd*` | `python scripts/train.py opd DOMAIN --method vector` |
| `gen_teacher_rollouts*` | `python scripts/train.py generate DOMAIN` |
| `*_distill_offline*` | `python scripts/train.py offpolicy DOMAIN --offline-data FILE` |
| `*_seqbasis*` | add `--method sequential --vectors 8 --stage-steps 100` |
| `*gated*` | add `--method gated --gate-rank 1` |

`DOMAIN` is `math`, `science`, `code`, or `instruction`. Existing offline distillation
uses teacher-generated trajectories with the historical reverse-KL surrogate; this
behavior is preserved and is not relabeled as supervised NLL.
