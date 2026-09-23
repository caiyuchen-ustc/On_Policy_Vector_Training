import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Keep the mathematical tests usable without importing the full GPU stack.
spec = importlib.util.spec_from_file_location(
    "alpha_stabler", Path(__file__).parents[2] / "verl/utils/alpha_stabler.py"
)
alpha = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = alpha
spec.loader.exec_module(alpha)


def test_projection_preserves_complement_and_removes_principal():
    torch.manual_seed(7)
    basis = torch.linalg.qr(torch.randn(12, 3, dtype=torch.float64)).Q
    gradient = torch.randn(2, 5, 12, dtype=torch.float64)
    projected = alpha.project_gradient(gradient, basis)
    assert torch.allclose(projected @ basis, torch.zeros(2, 5, 3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(projected + (gradient @ basis) @ basis.T, gradient)
    assert torch.allclose(alpha.project_gradient(projected, basis), projected)


def test_psi_sums_energy_before_averaging():
    # Opposite shifts cancel in their mean, but not in PSI.
    shift = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    energy = alpha.shift_energies(shift, torch.tensor([[1.0], [0.0]]))
    assert energy.tolist() == [2.0, 10.0, 2.0]


def controller():
    cfg = alpha.AlphaStablerConfig(
        warmup_steps=6, monitor_interval=1, ema_beta=0.0, persistence=2, principal_fraction=0.25
    )
    result = alpha.AlphaStabler(cfg, [0])
    torch.manual_seed(11)
    base = torch.randn(100, 4) * torch.tensor([20.0, 1.0, 0.5, 0.2])
    for i in range(6):
        shift = torch.tensor([0.1 + i * 0.005, 1.0, 0.0, 0.0]).expand_as(base)
        result.observe_update({0: base}, {0: base + shift})
    return result, base


def test_warmup_persistence_release_and_resume():
    ctl, base = controller()
    assert ctl.calibrated
    assert not ctl.predictor[0]["active"]
    ctl.observe_update({0: base}, {0: base + torch.tensor([3.0, 0.0, 0.0, 0.0])})
    assert not ctl.predictor[0]["active"]
    checkpoint = copy.deepcopy(ctl.state_dict())
    resumed = alpha.AlphaStabler(ctl.config, [0])
    resumed.load_state_dict(checkpoint)
    for obj in (ctl, resumed):
        obj.observe_update({0: base}, {0: base + torch.tensor([3.0, 0.0, 0.0, 0.0])})
        assert obj.predictor[0]["active"]
        obj.observe_update({0: base}, {0: base + torch.tensor([0.0, 0.0, 0.0, 1.0])})
        assert not obj.predictor[0]["active"]
    assert ctl.predictor == resumed.predictor


def test_invalid_shift_retains_ema_flag_resets_count():
    ctl, base = controller()
    ctl.predictor[0].update(active=True, count=2)
    previous = ctl.predictor[0]["ema"]
    ctl.observe_update({0: base}, {0: base})
    assert ctl.predictor[0]["active"]
    assert ctl.predictor[0]["ema"] == previous
    assert ctl.predictor[0]["count"] == 0


def test_no_shift_calibration_fails():
    ctl = alpha.AlphaStabler(alpha.AlphaStablerConfig(warmup_steps=3, monitor_interval=1), [0])
    base = torch.randn(8, 4)
    with pytest.raises(RuntimeError, match="no valid warm-up shifts"):
        for _ in range(3):
            ctl.observe_update({0: base}, {0: base})


def test_hook_changes_backward_only():
    ctl, _ = controller()
    ctl.predictor[0]["active"] = True
    layer = torch.nn.Linear(4, 4, bias=False)
    handle = layer.register_forward_hook(ctl.backward_hook(0))
    x = torch.randn(2, 4, requires_grad=True)
    output = layer(x)
    assert torch.equal(output, torch.nn.functional.linear(x, layer.weight))
    incoming = torch.randn_like(output)
    output.backward(incoming)
    expected = alpha.project_gradient(incoming, ctl.bases[0]) @ layer.weight
    assert torch.allclose(x.grad, expected, atol=1e-6)
    handle.remove()


def test_resume_during_warmup():
    cfg = alpha.AlphaStablerConfig(warmup_steps=6, monitor_interval=1)
    ctl = alpha.AlphaStabler(cfg, [0])
    base = torch.randn(20, 6)
    ctl.observe_update({0: base}, {0: base + 0.1})
    resumed = alpha.AlphaStabler(cfg, [0])
    resumed.load_state_dict(copy.deepcopy(ctl.state_dict()))
    assert resumed.step == 1
    assert torch.equal(resumed.records[0][0], ctl.records[0][0])
    assert torch.equal(resumed.moments[0][2], ctl.moments[0][2])


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden).tanh()
        return hidden


class TinyPolicy:
    ulysses_sequence_parallel_size = 1
    use_remove_padding = False

    def __init__(self, model):
        self.actor_module = model

    def _forward_micro_batch(self, inputs, **kwargs):
        return self.actor_module(inputs["hidden"])


class TinyBatch:
    def __init__(self, values):
        self.batch = values
        self.non_tensor_batch = {}

    def to(self, device):
        return self


def test_runtime_padding_gradient_accumulation_and_fixed_flags():
    torch.manual_seed(24)
    actor = TinyPolicy(TinyDecoder())
    reference = TinyPolicy(copy.deepcopy(actor.actor_module))
    with torch.no_grad():
        actor.actor_module.layers[0].bias.add_(0.1)
    cfg = alpha.AlphaStablerConfig(layers=[0, 1, 2], warmup_steps=3, monitor_interval=1, max_tokens_per_step=10)
    runtime = alpha.AlphaStablerRuntime(actor, reference, cfg)
    batches = [
        TinyBatch({"hidden": torch.randn(2, 5, 4), "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])})
        for _ in range(2)
    ]
    runtime.before_update(batches, 1.0, "cpu")
    assert runtime.controller.step == 1
    assert runtime.controller.moments[0][0] == 10
    assert actor.actor_module.training
    assert not reference.actor_module.training
    assert not any(parameter.grad is not None for parameter in actor.actor_module.parameters())
    runtime.controller.bases = {layer: torch.eye(4)[:, :1] for layer in cfg.layers}
    runtime.controller.predictor = {layer: {"active": True} for layer in cfg.layers}
    for batch in batches:
        actor._forward_micro_batch(batch.batch).square().mean().backward()
    assert runtime.controller.step == 1  # not a microbatch counter
    assert torch.equal(actor.actor_module.layers[0].weight.grad[0], torch.zeros(4))
    assert torch.isfinite(actor.actor_module.layers[1].weight.grad).all()


def test_hook_is_reinstalled_during_checkpoint_recomputation():
    from torch.utils.checkpoint import checkpoint

    ctl, _ = controller()
    ctl.predictor[0]["active"] = True
    layer = torch.nn.Linear(4, 4, bias=False)
    layer.register_forward_hook(ctl.backward_hook(0))
    x = torch.randn(3, 4, requires_grad=True)
    checkpoint(layer, x, use_reentrant=True).sum().backward()
    expected = alpha.project_gradient(torch.ones(3, 4), ctl.bases[0]) @ layer.weight
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_tuple_output_hook():
    ctl, _ = controller()
    ctl.predictor[0]["active"] = True
    x = torch.randn(2, 4, requires_grad=True)
    output = (x * 2, None)
    actual = ctl.backward_hook(0)(None, None, output)
    assert actual is output
    actual[0].sum().backward()
    expected = 2 * alpha.project_gradient(torch.ones_like(x), ctl.bases[0])
    assert torch.allclose(x.grad, expected)


def _distributed_worker(rank, store_path):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="file://" + store_path, rank=rank, world_size=2)
    try:
        cfg = alpha.AlphaStablerConfig(warmup_steps=3, monitor_interval=1, ema_beta=0.0, persistence=2)
        ctl = alpha.AlphaStabler(cfg, [0])
        torch.manual_seed(18 + rank)
        base = torch.randn(24, 4) * torch.tensor([12.0, 1.0, 0.5, 0.2])
        for step in range(3):
            # One worker has invalid/zero local shifts. The pooled observation
            # is valid and calibration/control must still agree on both workers.
            delta = torch.tensor([0.1 + step * 0.005, 1.0, 0.0, 0.0]) if rank else torch.zeros(4)
            ctl.observe_update({0: base}, {0: base + delta})
        shared = [None, None]
        dist.all_gather_object(shared, ctl.predictor)
        assert shared[0] == shared[1]
        bases = [torch.zeros_like(ctl.bases[0]) for _ in range(2)]
        dist.all_gather(bases, ctl.bases[0])
        assert torch.equal(bases[0], bases[1])
        for _ in range(2):
            delta = torch.tensor([8.0, 0.0, 0.0, 0.0]) if rank else torch.zeros(4)
            metrics = ctl.observe_update({0: base}, {0: base + delta})
        assert ctl.predictor[0]["active"]
        assert metrics["alpha_stabler/psi"] > 0.9
    finally:
        dist.destroy_process_group()


def test_two_rank_pooling_and_synchronized_control(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_worker, args=(str(tmp_path / "gloo-store"),), nprocs=2, join=True)
