# Alpha-Stabler

`verl/utils/alpha_stabler.py` implements the controller and the FSDP integration.
It is opt-in and adds no learned parameters. Forward activations, rollout generation,
rewards, and the existing RL loss are unchanged.

```bash
python scripts/train.py alpha science --method full --steps 2000
python scripts/train.py alpha instruction --method lora --lr 3e-5 --steps 2000
python scripts/train.py alpha science --monitor-only --name science-monitor
```

## Monitoring and control

1. During 50 optimizer updates, collect frozen-base activation covariance and
   actor-minus-base shifts on identical sampled valid token positions. Monitor
   decoder outputs at quarter, half, and three-quarter depth.
2. Freeze the top `ceil(0.10 * hidden_size)` principal basis per layer. Calibrate
   the warning/release thresholds from the warm-up median and scaled MAD. Empty
   or invalid calibration raises an error; thresholds are never silently replaced.
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

## Implementation choices and limits

- This implementation uses a detached **monitoring prepass** for the complete
  optimizer minibatch, followed by ordinary gradient accumulation. This avoids
  retaining every microbatch graph at once, makes all-rank decisions consistent,
  and keeps flags fixed during checkpoint recomputation. It adds an actor forward
  pass and a frozen-base forward pass on monitoring updates; no sub-2% overhead
  claim is made for this implementation.
- The default 256 sampled tokens per worker/update bound CPU calibration storage.
  Covariance accumulation uses float64 and centered second moments. Bases and
  gradient projection use float32. Increase the sampling budget for more stable
  estimates. PSI values are fractions in `[0, 1]`, not percentages.
- The reference is the actor's **initial base**, not an RL teacher. The launcher
  sets `model.base_model_path=model.path`; the worker rejects a mismatch. Frozen
  base weights are offloaded between monitoring passes.
- Supported integration: synchronous, text-only Qwen/DeepSeek-style FSDP workers,
  full or LoRA RL, Ulysses sequence parallelism 1. CPU tests verify the controller,
  and a tiny two-H20 FSDP smoke test verifies accumulated backward and CPU-offloaded
  reference integration. Actual LLM training and long-run stabilization remain to
  be measured. This is not implemented in the Megatron/new-engine worker paths.
- `alpha_stabler_rank_N.pt` is saved beside each actor checkpoint, including
  calibration state when resuming during warm-up. Resume with the same worker
  count, layer selection, controller configuration, and a persistent shared local
  checkpoint directory. Include these files if copying a checkpoint elsewhere.

Metrics include `alpha_stabler/psi`, `alpha_stabler/active_layers`, and per-layer
PSI, EMA, warning/release thresholds, and flags. See `analysis/README.md` for plotting.
