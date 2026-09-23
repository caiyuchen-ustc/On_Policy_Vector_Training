"""Principal-subspace intrusion monitoring and backward-only activation control.

The controller has no trainable parameters and never changes forward activations.
Statistics and calibration records live on CPU. Distributed workers pool sufficient
statistics and shift energies before making the same threshold/activation decisions.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass
class AlphaStablerConfig:
    enabled: bool = False
    layers: list[int] | None = None
    principal_fraction: float = 0.10
    warmup_steps: int = 50
    monitor_interval: int = 3
    ema_beta: float = 0.95
    persistence: int = 3
    warning_mad: float = 3.0
    release_mad: float = 2.0
    epsilon: float = 1e-8
    min_shift_energy: float = 1e-10
    max_tokens_per_step: int = 256
    control: bool = True

    def __post_init__(self):
        if not 0 < self.principal_fraction < 1:
            raise ValueError("principal_fraction must be in (0, 1)")
        if self.warmup_steps < self.monitor_interval or self.monitor_interval < 1:
            raise ValueError("warmup_steps must include at least one monitoring interval")
        if not 0 <= self.ema_beta < 1 or self.persistence < 1:
            raise ValueError("Invalid EMA coefficient or persistence")
        if not 0 <= self.release_mad < self.warning_mad:
            raise ValueError("Require 0 <= release_mad < warning_mad")
        if self.epsilon <= 0 or self.min_shift_energy < 0 or self.max_tokens_per_step < 1:
            raise ValueError("Invalid numerical floor or token budget")


def distributed_sum(value):
    if not dist.is_available() or not dist.is_initialized():
        return value
    original = value.device
    if dist.get_backend() == "nccl":
        value = value.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.to(original)


def project_gradient(gradient, basis):
    """Remove only the component in span(basis), using fp32 accumulation."""
    dtype = gradient.dtype
    work = gradient.float() if dtype in (torch.float16, torch.bfloat16) else gradient
    basis = basis.to(device=work.device, dtype=work.dtype)
    return (work - (work @ basis) @ basis.T).to(dtype)


def shift_energies(shift, basis):
    shift = shift.to(dtype=torch.float64)
    basis = basis.to(device=shift.device, dtype=shift.dtype)
    return torch.stack([(shift @ basis).square().sum(), shift.square().sum(), shift.new_tensor(shift.shape[0])])


class AlphaStabler:
    def __init__(self, config, layers):
        self.config = config
        self.layers = list(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Monitored layers must be nonempty and unique")
        self.step = 0
        self.bases = {}
        self.moments = {}
        self.records = {layer: [] for layer in self.layers}
        self.predictor = {}
        self._device_bases = {}

    @property
    def calibrated(self):
        return bool(self.bases)

    def needs_observation(self):
        next_step = self.step + 1
        return next_step <= self.config.warmup_steps or next_step % self.config.monitor_interval == 0

    def _valid(self, energy):
        return bool(
            torch.isfinite(energy).all() and energy[2] > 0 and energy[1] / energy[2] > self.config.min_shift_energy
        )

    def observe_update(self, reference=None, current=None):
        """Call exactly once before each optimizer update (not once per rollout)."""
        self.step += 1
        metrics = {"alpha_stabler/update": self.step}
        if reference is None or current is None:
            return metrics
        check = self.step % self.config.monitor_interval == 0
        if self.step <= self.config.warmup_steps:
            for layer in self.layers:
                base = reference[layer].double().cpu()
                actor = current[layer].double().cpu()
                if base.shape != actor.shape or base.ndim != 2:
                    raise ValueError(f"Mismatched base/actor activations at layer {layer}")
                if not torch.isfinite(base).all() or not torch.isfinite(actor).all():
                    raise ValueError("Non-finite warm-up activations; recalibration required")
                n, width = base.shape
                if layer not in self.moments:
                    self.moments[layer] = [
                        torch.zeros((), dtype=torch.float64),
                        torch.zeros(width, dtype=torch.float64),
                        torch.zeros(width, width, dtype=torch.float64),
                    ]
                moments = self.moments[layer]
                moments[0] += n
                moments[1] += base.sum(0)
                moments[2] += base.T @ base
                if check:
                    # Keep all ranks' record positions aligned, including zero shifts.
                    self.records[layer].append((actor - base).float())
            if self.step == self.config.warmup_steps:
                self.calibrate()
            return metrics
        if not self.calibrated:
            raise RuntimeError("Missing Alpha-Stabler calibration; warm-up observations are required")
        if not check:
            return metrics
        total = torch.zeros(3, dtype=torch.float64)
        for layer in self.layers:
            delta = current[layer].double().cpu() - reference[layer].double().cpu()
            energy = distributed_sum(shift_energies(delta, self.bases[layer]))
            state = self.predictor[layer]
            if self._valid(energy):
                psi = float(energy[0] / (energy[1] + self.config.epsilon))
                self.update_predictor(layer, psi)
                total += energy
                metrics[f"alpha_stabler/layer_{layer}/psi"] = psi
            else:
                state["count"] = 0
            metrics.update(
                {
                    f"alpha_stabler/layer_{layer}/{key}": float(state[key])
                    for key in ("ema", "active", "warning", "release")
                }
            )
        if self._valid(total):
            metrics["alpha_stabler/psi"] = float(total[0] / (total[1] + self.config.epsilon))
        metrics["alpha_stabler/active_layers"] = sum(s["active"] for s in self.predictor.values())
        return metrics

    def calibrate(self):
        bases, predictors = {}, {}
        for layer in self.layers:
            n, sums, second = [distributed_sum(t.clone()) for t in self.moments[layer]]
            if n < 2:
                raise RuntimeError(f"Layer {layer}: insufficient reference tokens for PCA")
            covariance = (second - torch.outer(sums, sums) / n) / (n - 1)
            _, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
            rank = math.ceil(self.config.principal_fraction * vectors.shape[0])
            basis = vectors[:, -rank:].float().contiguous()
            # Eigenvector signs/order in degenerate eigenspaces must agree across ranks.
            if dist.is_available() and dist.is_initialized():
                device = (
                    torch.device("cuda", torch.cuda.current_device())
                    if dist.get_backend() == "nccl"
                    else torch.device("cpu")
                )
                shared = basis.to(device)
                dist.broadcast(shared, src=0)
                basis = shared.cpu()
            values = []
            for shift in self.records[layer]:
                energy = distributed_sum(shift_energies(shift, basis))
                if self._valid(energy):
                    values.append(float(energy[0] / (energy[1] + self.config.epsilon)))
            if not values:
                raise RuntimeError(f"Layer {layer}: no valid warm-up shifts; increase warmup_steps or recalibrate")
            values = torch.tensor(values, dtype=torch.float64)
            median = float(values.quantile(0.5))
            scale = 1.4826 * float((values - median).abs().quantile(0.5)) + self.config.epsilon
            warning = median + self.config.warning_mad * scale
            release = median + self.config.release_mad * scale
            if not 0 < release < warning < 1:
                raise RuntimeError(f"Layer {layer}: invalid calibration thresholds ({release}, {warning}); recalibrate")
            bases[layer] = basis
            predictors[layer] = {"ema": median, "warning": warning, "release": release, "count": 0, "active": False}
        self.bases, self.predictor = bases, predictors
        self.records, self.moments = {}, {}

    def update_predictor(self, layer, psi):
        state = self.predictor[layer]
        state["ema"] = self.config.ema_beta * state["ema"] + (1 - self.config.ema_beta) * psi
        state["count"] = state["count"] + 1 if state["ema"] > state["warning"] else 0
        if state["count"] >= self.config.persistence:
            state["active"] = True
        if state["ema"] < state["release"]:
            state["active"] = False

    def backward_hook(self, layer):
        def forward_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            active = self.config.control and self.calibrated and self.predictor[layer]["active"]
            if active and torch.is_grad_enabled() and hidden.requires_grad:
                key = (layer, str(hidden.device))
                if key not in self._device_bases:
                    self._device_bases[key] = self.bases[layer].to(hidden.device)
                basis = self._device_bases[key]
                # The closure snapshots the flag/basis for this forward/backward pair.
                hidden.register_hook(lambda gradient: project_gradient(gradient, basis))
            return output

        return forward_hook

    def state_dict(self):
        return {
            "version": 1,
            "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
            "config": asdict(self.config),
            "layers": self.layers,
            "step": self.step,
            "bases": self.bases,
            "predictor": self.predictor,
            "moments": self.moments,
            "records": self.records,
        }

    def load_state_dict(self, state):
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if state.get("world_size", 1) != world_size:
            raise ValueError("Alpha-Stabler resume requires the original data-parallel world size")
        if state["version"] != 1 or state["layers"] != self.layers or state["config"] != asdict(self.config):
            raise ValueError("Alpha-Stabler checkpoint/config mismatch")
        for key in ("step", "bases", "predictor", "moments", "records"):
            setattr(self, key, state[key])
        self._device_bases.clear()


def decoder_layers(module):
    """Find Qwen/DeepSeek decoder layers through PEFT and FSDP wrappers."""
    candidates = [
        (name, child)
        for name, child in module.named_modules()
        if name.endswith("layers") and isinstance(child, torch.nn.ModuleList)
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one decoder layer stack, found {[n for n, _ in candidates]}")
    return candidates[0][1]


class AlphaStablerRuntime:
    """Adapter for the FSDP actor's existing microbatch forward function.

    Monitoring uses a detached prepass across the entire optimization minibatch so
    flags stay constant across gradient accumulation and checkpoint recomputation.
    The extra actor prepass trades compute for bounded memory; measure its overhead.
    """

    def __init__(self, actor, reference, config, load_reference=None, offload_reference=None):
        if actor.ulysses_sequence_parallel_size != 1:
            raise ValueError("Alpha-Stabler currently supports FSDP data parallelism with Ulysses size 1")
        actor_layers, reference_layers = decoder_layers(actor.actor_module), decoder_layers(reference.actor_module)
        if len(actor_layers) != len(reference_layers):
            raise ValueError("Alpha-Stabler requires the actor's own frozen initial base architecture")
        selected = config.layers or sorted({max(0, math.ceil(len(actor_layers) * f) - 1) for f in (0.25, 0.5, 0.75)})
        if any(i < 0 or i >= len(actor_layers) for i in selected):
            raise ValueError("Alpha-Stabler layer index outside decoder")
        self.controller = AlphaStabler(config, selected)
        self.actor, self.reference = actor, reference
        if getattr(actor, "use_fused_kernels", False) or getattr(reference, "use_fused_kernels", False):
            raise ValueError("Alpha-Stabler requires unfused decoder outputs for its hooks")
        self.load_reference, self.offload_reference = load_reference, offload_reference
        self.collecting = None
        self.mask = None
        self.capture = {}
        self.handles = []
        for index in selected:
            self.handles.append(actor_layers[index].register_forward_hook(self.controller.backward_hook(index)))
            self.handles.append(actor_layers[index].register_forward_hook(self._capture_hook(index, "actor")))
            self.handles.append(reference_layers[index].register_forward_hook(self._capture_hook(index, "reference")))

    def _capture_hook(self, layer, role):
        def hook(_module, _inputs, output):
            if self.collecting != role:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            flat = hidden.detach().reshape(-1, hidden.shape[-1])
            mask = self.mask.reshape(-1).bool()
            policy = self.actor if role == "actor" else self.reference
            if not policy.use_remove_padding:
                flat = flat[mask]
            elif flat.shape[0] != int(mask.sum()):
                raise ValueError("Packed activations do not match valid token count")
            # Spread the bounded sample over the full microbatch, never select padding.
            count = min(self.remaining, flat.shape[0])
            indices = torch.linspace(0, max(flat.shape[0] - 1, 0), count, device=flat.device).long()
            self.capture[layer] = flat[indices].float().cpu()

        return hook

    @torch.no_grad()
    def before_update(self, micro_batches, temperature, device):
        controller = self.controller
        if not controller.needs_observation():
            return controller.observe_update()
        collected = {role: {layer: [] for layer in controller.layers} for role in ("actor", "reference")}
        self.remaining = controller.config.max_tokens_per_step
        was_training = self.actor.actor_module.training
        self.actor.actor_module.eval()
        self.reference.actor_module.eval()
        try:
            if self.load_reference:
                self.load_reference()
            # Every rank runs all microbatches: FSDP collectives must remain aligned.
            for batch in micro_batches:
                batch = batch.to(device)
                inputs = {**batch.batch, **batch.non_tensor_batch}
                self.mask = inputs["attention_mask"]
                for role, policy in (("reference", self.reference), ("actor", self.actor)):
                    self.collecting, self.capture = role, {}
                    policy._forward_micro_batch(inputs, temperature=temperature, calculate_entropy=False)
                    for layer in controller.layers:
                        collected[role][layer].append(self.capture[layer])
                self.remaining -= self.capture[controller.layers[0]].shape[0]
            base = {k: torch.cat(v) for k, v in collected["reference"].items()}
            actor = {k: torch.cat(v) for k, v in collected["actor"].items()}
            return controller.observe_update(base, actor)
        finally:
            self.collecting, self.capture, self.mask = None, {}, None
            self.actor.actor_module.train(was_training)
            if self.offload_reference:
                self.offload_reference()
