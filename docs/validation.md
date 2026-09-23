# Testing

## Commands

```bash
python -m pytest -q tests/opv
python scripts/check_repository.py
python scripts/data/verify_hub.py
python scripts/smoke_alpha_fsdp.py --gpus 2
```

## Coverage

- Tests cover gradient projection, token-level PSI, calibration, trigger/release
  behavior, invalid observations, checkpoint state, gradient accumulation and
  recomputation, distributed synchronization, byte-preserving dataset copies,
  direct launcher invocation and Tensor/NumPy metric serialization.
- 76 training configurations compose and resolve with Hydra; launcher Shell syntax
  and Python syntax checks pass.
- All 11 published Parquet files match the original files and their Hugging Face
  Git/LFS hashes at the revisions in `configs/datasets.json`.
- A two-H20 FSDP test completes eight tiny-model updates with a CPU-offloaded
  reference, accumulated backward projection and state reload. It forces control
  active after calibration to exercise the projection hooks; automatic triggering
  is covered by the unit tests.
- DeepSeek-R1-Distill-Qwen-1.5B completed seven GRPO updates on two H20 GPUs using
  SciKnowEval, synchronous vLLM and the FSDP actor. Six warm-up updates calibrated
  the principal bases; update seven recorded PSI. This startup check used four
  prompts, two responses per prompt, 32 response tokens and an entropy coefficient
  of `0.001`. It exited successfully with finite gradients and JSON metrics.

The seven-step run validates startup, updates and calibration; long-run stabilization
and runtime overhead require separate experiments.

## Short training run

```bash
python scripts/train.py alpha science --name alpha-smoke --gpus 8 --tp 2 \
  --batch-size 8 --mini-batch 8 --samples 2 --max-prompt 512 --max-response 64 \
  --steps 12 --eval-every 6 --val-max-samples 8 --save-every 6 \
  actor_rollout_ref.actor.alpha_stabler.warmup_steps=6 \
  actor_rollout_ref.actor.alpha_stabler.monitor_interval=1
```

If calibration reports insufficient shifts or invalid thresholds, increase warm-up
duration or token sampling. Use the default settings for full experiments.
