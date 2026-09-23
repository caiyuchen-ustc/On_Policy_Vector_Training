# Alpha-Stabler

`verl/utils/alpha_stabler.py` implements the controller and the FSDP integration.
It is opt-in and adds no learned parameters. Forward activations, rollout generation,
rewards, and the existing RL loss are unchanged.

```bash
bash scripts/alpha_stabler/run.sh
bash scripts/alpha_stabler/run.sh instruction --method lora --lr 3e-5 --steps 2000
bash scripts/alpha_stabler/run.sh science --monitor-only --name science-monitor
```

With no arguments, the launcher trains DeepSeek-R1-Distill-Qwen-1.5B on SciKnowEval
using 8 GPUs. Missing model and dataset files are downloaded from Hugging Face;
existing local files are reused without conversion. Choose another task with
`math`, `code`, or `instruction`, and adjust `--gpus` / `--tp` for your machine.

```bash
# Validate dependencies, GPU visibility, model files and dataset schemas.
bash scripts/alpha_stabler/run.sh --check
# Use an existing local model and require all files to be present.
bash scripts/alpha_stabler/run.sh --model /path/to/model --no-download
# Preview the configuration without loading a model or downloading files.
bash scripts/alpha_stabler/run.sh --config-only
```

Set `PYTHON=/path/to/python` to select a training environment. `--ray-cpus` controls
the CPU allocation for the local Ray runtime. Use `--train-file` / `--val-files` for
custom data paths. All settings also accept Hydra `key=value` overrides. The alpha
launcher defaults to the vLLM V0 synchronous engine and eager execution.

## Monitoring and control

1. During 50 optimizer updates, collect frozen-base activation covariance and
   actor-minus-base shifts on identical sampled valid token positions. Monitor
   decoder outputs at quarter, half, and three-quarter depth.
2. Freeze the top `ceil(0.10 * hidden_size)` principal basis per layer. Calibrate
   the warning/release thresholds from the warm-up median and scaled MAD.
   Empty or invalid calibration raises an error and requires recalibration.
3. Every 3 optimizer updates, sum token-level principal/total shift energies and
   reduce them across data-parallel workers. Update a `.95` EMA. Three consecutive
   warnings enable control; falling below the lower release threshold disables it.
4. For enabled layers, replace each incoming activation gradient by
   `G - (G @ U) @ U.T`. The complementary component is preserved. Flags remain fixed
   across the entire accumulation/backward/optimizer update.

All settings are under `actor_rollout_ref.actor.alpha_stabler`. Example:

```bash
python scripts/train.py alpha science \
  actor_rollout_ref.actor.alpha_stabler.layers='[6,13,20]' \
  actor_rollout_ref.actor.alpha_stabler.max_tokens_per_step=512 \
  actor_rollout_ref.actor.alpha_stabler.warmup_steps=60
```

## Runtime configuration

- A detached monitoring pass over the optimizer minibatch precedes gradient
  accumulation. Control flags stay fixed during backward and checkpoint
  recomputation. Each monitoring update adds an actor forward pass and a frozen-base
  forward pass; the monitoring interval controls this additional work.
- The default 256 sampled tokens per worker/update bound CPU calibration storage.
  Covariance accumulation uses float64 and centered second moments. Bases and
  gradient projection use float32. Increase the sampling budget for more stable
  estimates. PSI values are fractions in `[0, 1]`.
- The reference is the actor's frozen initial base. The launcher sets
  `model.base_model_path=model.path`; the worker rejects a mismatch. Frozen
  base weights are offloaded between monitoring passes.
- Supported integration: synchronous, text-only Qwen/DeepSeek-style FSDP workers,
  full or LoRA RL, Ulysses sequence parallelism 1. See [testing](validation.md)
  for the validation coverage.
- `alpha_stabler_rank_N.pt` is saved beside each actor checkpoint, including
  calibration state when resuming during warm-up. Resume with the same worker
  count, layer selection, controller configuration, and a persistent shared local
  checkpoint directory. Include these files if copying a checkpoint elsewhere.

Metrics include `alpha_stabler/psi`, `alpha_stabler/active_layers`, and per-layer
PSI, EMA, warning/release thresholds, and flags. See `analysis/README.md` for plotting.
