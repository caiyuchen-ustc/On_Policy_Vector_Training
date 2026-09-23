# Validation and scope

This release organizes the original experiments and adds an opt-in Alpha-Stabler
implementation. The existing datasets, train/evaluation splits, prompts and reward
formats were preserved as requested. No MATH500 or replacement split was added.

## Completed checks

- All 11 published Parquet files match the source files byte-for-byte (SHA-256).
  The hashes were additionally compared with Hugging Face Git/LFS object metadata
  at the immutable revisions in `configs/datasets.json`.
- Four teacher repositories and their revision IDs were verified through the Hub.
- 76 launcher combinations compose and resolve successfully with Hydra, including
  four tasks, RL, OPD, offline distillation, generation, Alpha-Stabler, and DAPO.
- Shell syntax checked for retained launchers; all Python source parses. New Python
  files pass Ruff checks. Credential scanning covers released script/analysis paths.
- 12 CPU tests passed, covering projection orthogonality and complement preservation, uncancelled
  token-level PSI, calibration, activation persistence/release, invalid observations,
  warm-up and calibrated resume, gradient accumulation, checkpoint recomputation,
  tuple outputs, and unchanged dataset bytes.
- A two-process Gloo test verifies pooled calibration, identical PCA bases and
  synchronized control, including a worker with zero local shift.
- Combined code evaluation data loads with Hugging Face Datasets (1,904 rows).
  The FSDP actor configuration instantiates with the new controller settings.
- `scripts/smoke_alpha_fsdp.py --gpus 2` passed on two H20 GPUs: a tiny model
  completed eight updates with FSDP sharding, CPU-offloaded reference weights,
  monitoring calibration, accumulated backward projection and state reload.
  Control is deliberately forced active after calibration in this integration test;
  automatic triggering/release is tested separately in the CPU suite.

## Not measured here

CUDA was inaccessible inside the sandbox; the authorized smoke test ran outside it.
No pretrained LLM training, vLLM rollout, or 2,000-step stabilization experiment was
run. The tiny FSDP test and CPU/configuration checks do not establish convergence or
the manuscript's claimed speed/performance. Alpha-Stabler's monitoring prepass has
additional work; wall time and peak memory need profiling with the actual LLMs.

Historical `make_*.py` analysis scripts and supplied plots include items explicitly
marked illustrative or adjusted in the original analysis notes. They are retained
with that provenance. `analysis/plot_metrics.py` plots actual new-run logs only.

For a GPU smoke run, use a distinct output name and a small prompt/token budget, e.g.:

```bash
python scripts/train.py alpha science --name alpha-smoke --gpus 8 --tp 2 \
  --batch-size 8 --mini-batch 8 --samples 2 --max-prompt 512 --max-response 64 \
  --steps 12 --eval-every 6 --val-max-samples 8 --save-every 6 \
  actor_rollout_ref.actor.alpha_stabler.warmup_steps=6 \
  actor_rollout_ref.actor.alpha_stabler.monitor_interval=1
```

Short smoke calibration can legitimately fail if shifts do not produce valid
thresholds. Increase warm-up/token sampling and recalibrate; do not replace failed
calibration with arbitrary thresholds. Use the full defaults for actual experiments.
