# Analysis

The existing analysis scripts, caches, and figures are retained. New training runs
write `outputs/<run>/metrics.jsonl` through verl's file logger; W&B is optional.

## Training curves and Alpha-Stabler

```bash
python analysis/plot_metrics.py outputs/science-alpha-full/metrics.jsonl --list-keys
python analysis/plot_metrics.py outputs/science-alpha-full/metrics.jsonl \
  --keys alpha_stabler/psi alpha_stabler/active_layers \
  --output analysis/figures/science-alpha.png
```

Pass multiple logs plus `--labels` to compare runs. Use the validation metric key
printed by `--list-keys` to add a reward/accuracy panel. Missing measurements are
errors; this entry point does not synthesize continuations or interpolate scores.

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

For each legacy script, inspect its arguments and input paths before running it.
Checkpoint-dependent analysis requires the corresponding trained vectors; the
repository does not contain those large model artifacts. The portable activation
extractor accepts `--model`, `--data`, `--layers`, `--output`, and `--device`.

Historical scripts resolve paths through `vector_study/_paths.py`: set
`OPV_MODEL_ROOT` for local Hugging Face checkpoints (some scripts open safetensors
directly), `OPV_ARTIFACT_ROOT` for the historical vector/checkpoint directory tree,
and `OPV_DATA_ROOT` for downloaded data. Output tables/figures use this repository's
analysis directory. Domain-specific scripts may still require the existing
per-domain files under `sciknoweval/by_domain`; no data is repartitioned automatically.

## Figure provenance

`vector_study/README.md` contains the original per-figure provenance notes. Several
legacy plotting scripts intentionally contain illustrative values, adjusted curves,
or placeholders. In particular `make_psi_alpha_stabler*.py` and
`make_alpha_stabler_ablations.py` are illustration scripts, **not** the Alpha-Stabler
training implementation or measurements from the new implementation. Their existing
outputs are preserved as historical assets. Use `plot_metrics.py` for measured
results from new runs. No empirical stabilization claim is established by CPU tests.
