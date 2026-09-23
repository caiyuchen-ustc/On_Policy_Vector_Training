# Analysis

Training runs write `outputs/<run>/metrics.jsonl` through verl's file logger.
This directory contains plotting scripts, representation analyses and figure assets.

## Training curves and Alpha-Stabler

```bash
python analysis/plot_metrics.py outputs/science-alpha-full/metrics.jsonl --list-keys
python analysis/plot_metrics.py outputs/science-alpha-full/metrics.jsonl \
  --keys alpha_stabler/psi alpha_stabler/active_layers \
  --output analysis/figures/science-alpha.png
```

Pass multiple logs plus `--labels` to compare runs. Use a validation metric key
printed by `--list-keys` to add a reward/accuracy panel. Each requested metric must
be present in the input logs.

## Existing experiments

| Area | Directory / entry points |
| --- | --- |
| RL / OPD / off-policy curves | `opd_plots/plot_opd_score_curves.py`, cached `data_*.json` |
| Depth and gating ablations | `opd_plots/make_layer_*.py`, `make_gate_ablations.py` |
| Activation covariance / PCs | `vector_study/extract_activations.py` |
| Vector and principal-subspace geometry | `vector_study/hidden_subspace.py`, `seq_vector_hidden_position.py`, `project_all_domains.py` |
| Vocabulary readout | `vector_study/logit_lens.py`, `make_logitlens_fig.py` |
| Teacher/regime comparisons | `vector_study/crossrun_consistency.py`, `make_teacher_sim_bar.py`, `make_onoff_matrix.py` |
| Cross-domain directions | `vector_study/make_domain_transfer.py`, `make_domain_mix_heatmap.py` |

Checkpoint-dependent analysis takes the corresponding trained vectors as input.
The activation extractor accepts `--model`, `--data`, `--layers`, `--output`, and `--device`.

Historical scripts resolve paths through `vector_study/_paths.py`: set
`OPV_MODEL_ROOT` for local Hugging Face checkpoints (some scripts open safetensors
directly), `OPV_ARTIFACT_ROOT` for the historical vector/checkpoint directory tree,
and `OPV_DATA_ROOT` for downloaded data. Output tables/figures use this repository's
analysis directory. Domain-specific scripts use the existing per-domain files under
`sciknoweval/by_domain`.

## Plot inputs

`plot_metrics.py` plots recorded training metrics. `make_psi_alpha_stabler*.py` and
`make_alpha_stabler_ablations.py` generate illustrative curves containing synthetic
values, rather than measured Alpha-Stabler results. Per-figure sources and adjustments
are documented in [vector_study/README.md](vector_study/README.md) and script docstrings.
