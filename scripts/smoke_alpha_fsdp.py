#!/usr/bin/env python3
"""Tiny two-GPU FSDP smoke test for Alpha-Stabler (no pretrained downloads)."""

import argparse
import copy
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("alpha_stabler_smoke", ROOT / "verl/utils/alpha_stabler.py")
alpha = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = alpha
spec.loader.exec_module(alpha)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(4)])
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(16))
                layer.bias.zero_()

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class Policy:
    ulysses_sequence_parallel_size = 1
    use_remove_padding = False

    def __init__(self, model):
        self.actor_module = model

    def _forward_micro_batch(self, inputs, **_kwargs):
        return self.actor_module(inputs["hidden"])


class Batch:
    def __init__(self, hidden, mask):
        self.batch = {"hidden": hidden, "attention_mask": mask}
        self.non_tensor_batch = {}

    def to(self, device):
        return Batch(self.batch["hidden"].to(device), self.batch["attention_mask"].to(device))


def worker(rank, world_size, store_path):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="file://" + store_path, rank=rank, world_size=world_size)
    try:
        torch.manual_seed(33)
        base = Decoder()
        model = copy.deepcopy(base)
        with torch.no_grad():
            model.layers[0].bias[-1] = 0.1
        opts = dict(device_id=rank, use_orig_params=True, auto_wrap_policy=ModuleWrapPolicy({torch.nn.Linear}))
        actor = Policy(FSDP(model, **opts))
        reference = Policy(FSDP(base, cpu_offload=CPUOffload(offload_params=True), **opts))
        runtime = alpha.AlphaStablerRuntime(
            actor,
            reference,
            alpha.AlphaStablerConfig(warmup_steps=6, monitor_interval=1, max_tokens_per_step=128, persistence=2),
        )
        optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=1e-4)
        for step in range(8):
            torch.manual_seed(100 + step + rank * 100)
            scale = torch.ones(16)
            scale[0] = 10.0
            batches = [Batch(torch.randn(2, 32, 16) * scale, torch.ones(2, 32, dtype=torch.long)) for _ in range(2)]
            metrics = runtime.before_update(batches, 1.0, rank)
            if runtime.controller.calibrated:
                # Force activation here to exercise the backward hooks; trigger
                # calibration/hysteresis are covered separately by unit tests.
                for state in runtime.controller.predictor.values():
                    state["active"] = True
            optimizer.zero_grad()
            for batch in batches:
                hidden = batch.to(rank).batch
                output = actor._forward_micro_batch(hidden)
                # Calibrate a controlled complement-dominated update; the test
                # checks integration, not whether an arbitrary toy loss is stable.
                loss = (output[..., -1] - 1).square().mean() / len(batches)
                loss.backward()
            for parameter in actor.actor_module.parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()
            optimizer.step()
        assert runtime.controller.calibrated
        state = runtime.controller.state_dict()
        runtime.controller.load_state_dict(state)
        if rank == 0:
            print(f"PASS: {world_size}-GPU FSDP, CPU-offloaded base, 8 updates, accumulated backward, resume state")
            print(f"Final pooled PSI: {metrics.get('alpha_stabler/psi')}")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Requires {args.gpus} CUDA GPUs")
    with tempfile.TemporaryDirectory(prefix="alpha-fsdp-smoke-") as folder:
        mp.spawn(worker, args=(args.gpus, os.path.join(folder, "store")), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
