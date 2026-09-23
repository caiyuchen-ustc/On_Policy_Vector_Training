#!/usr/bin/env python3
"""Portable launchers for the existing RL / OPD / offline-distillation experiments."""

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEACHERS = json.loads((ROOT / "configs/teachers.json").read_text())


def value(item):
    if isinstance(item, (bool, list, dict)) or item is None:
        return json.dumps(item, separators=(",", ":"))
    if isinstance(item, str):
        return json.dumps(item)
    return str(item)


def build_command(args, extra):
    task = TEACHERS[args.domain]
    model = args.model or task["base"]
    teacher = args.teacher or task["teacher"]
    rl = args.stage in ("rl", "alpha")
    vector = args.method in ("vector", "sequential", "gated")
    data_dir = Path(args.data_dir).resolve() / args.domain
    train = args.train_file or str(
        data_dir / ("teacher_train.parquet" if rl and args.domain == "math" else "train.parquet")
    )
    vals = args.val_files or [str(data_dir / "validation.parquet")]
    if not args.val_files and args.domain == "math":
        vals.append(str(data_dir / "aime2025.parquet"))
    if not args.val_files and args.domain == "code":
        vals.append(str(data_dir / "eurus_validation.parquet"))
    prompt_len = args.max_prompt or (
        8192 if args.domain == "science" else 3072 if args.domain == "instruction" or not rl else 2048
    )
    response_len = args.max_response or (20480 if rl and args.domain == "instruction" else 16384)
    batch_size = args.batch_size or ((128 if args.domain == "math" else 256) if rl else 1024)
    mini_batch = args.mini_batch or (32 if rl else batch_size)
    default_lr = (
        (3e-5 if args.method == "lora" else 1e-6)
        if rl
        else (1e-1 if vector else 5e-6 if args.method == "lora" else 1e-5)
    )
    if args.stage == "offpolicy":
        default_lr = 5e-2 if vector else 1e-5 if args.method == "lora" else 1e-6
    lr = args.lr if args.lr is not None else default_lr
    run_name = args.name or f"{args.domain}-{args.stage}-{args.method}"
    output = Path(args.output_dir).resolve() / run_name
    parallel_size = args.tp or (4 if rl else 2)
    if args.gpus < 1 or args.nodes < 1 or parallel_size < 1 or args.gpus * args.nodes % parallel_size:
        raise ValueError("Total GPUs must be positive and divisible by rollout tensor parallel size")
    if batch_size < 1 or mini_batch < 1 or mini_batch > batch_size or lr <= 0:
        raise ValueError("Require positive learning rate and batch_size >= mini_batch > 0")
    if args.stage == "generate":
        generation_output = str(output / "round_0.parquet")
        opts = {
            "model.path": teacher,
            "data.path": train,
            "data.prompt_key": "prompt",
            "data.n_samples": args.samples or 1,
            "data.batch_size": args.batch_size or 128,
            "data.output_path": generation_output,
            "rollout.prompt_length": prompt_len,
            "rollout.response_length": response_len,
            "rollout.tensor_model_parallel_size": args.tp or 2,
            "rollout.temperature": 1.0,
            "rollout.top_p": 1.0,
            "rollout.seed": args.seed,
            "rollout.max_num_batched_tokens": 32768,
            "trainer.n_gpus_per_node": args.gpus,
            "trainer.nnodes": args.nodes,
        }
        if "qwen3" in teacher.lower():
            opts["+data.apply_chat_template_kwargs.enable_thinking"] = False
        module = "verl.trainer.main_generation"
    else:
        opts = {
            "algorithm.adv_estimator": "grpo",
            "algorithm.use_kl_in_reward": False,
            "data.train_files": train,
            "data.val_files": vals,
            "data.prompt_key": "prompt",
            "data.train_batch_size": batch_size,
            "data.max_prompt_length": prompt_len,
            "data.max_response_length": response_len,
            "data.filter_overlong_prompts": True,
            "data.truncation": "left",
            "data.shuffle": True,
            "data.seed": args.seed,
            "data.return_raw_chat": True,
            "data.val_max_samples": args.val_max_samples,
            "actor_rollout_ref.model.path": model,
            "actor_rollout_ref.model.use_remove_padding": True,
            "actor_rollout_ref.model.enable_gradient_checkpointing": True,
            "actor_rollout_ref.model.lora_rank": args.lora_rank if args.method == "lora" else 0,
            "actor_rollout_ref.model.lora_alpha": args.lora_rank * 2,
            "actor_rollout_ref.model.target_modules": "all-linear",
            "actor_rollout_ref.model.enable_trainable_token_vector": vector,
            "actor_rollout_ref.actor.optim.lr": lr,
            "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
            "actor_rollout_ref.actor.ppo_mini_batch_size": mini_batch,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
            "actor_rollout_ref.actor.use_dynamic_bsz": True,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 32768,
            "actor_rollout_ref.actor.entropy_coeff": 0.0,
            "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
            "actor_rollout_ref.actor.fsdp_config.use_orig_params": True,
            "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages": not rl,
            "actor_rollout_ref.actor.use_kl_loss": not rl or args.rl_algorithm == "grpo",
            "actor_rollout_ref.actor.kl_loss_coef": 0.001 if rl and args.rl_algorithm == "grpo" else 0.0,
            "actor_rollout_ref.rollout.name": "vllm",
            "actor_rollout_ref.rollout.mode": "sync",
            "actor_rollout_ref.rollout.n": args.samples or ((16 if args.domain == "math" else 8) if rl else 1),
            "actor_rollout_ref.rollout.tensor_model_parallel_size": args.tp or (4 if rl else 2),
            "actor_rollout_ref.rollout.gpu_memory_utilization": args.gpu_memory,
            "actor_rollout_ref.rollout.enforce_eager": vector,
            "actor_rollout_ref.rollout.free_cache_engine": True,
            "actor_rollout_ref.rollout.calculate_log_probs": True,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
            "actor_rollout_ref.rollout.max_num_batched_tokens": 32768,
            "actor_rollout_ref.rollout.temperature": 1.0,
            "actor_rollout_ref.rollout.top_p": 1.0,
            "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
            "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
            "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
            "actor_rollout_ref.rollout.val_kwargs.n": 4,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
            "actor_rollout_ref.ref.fsdp_config.param_offload": True,
            "actor_rollout_ref.ref.fsdp_config.use_orig_params": True,
            "reward_model.reward_manager": "naive",
            "trainer.project_name": "opv",
            "trainer.experiment_name": run_name,
            "trainer.logger": ["console", "file", "wandb"] if args.wandb else ["console", "file"],
            "trainer.n_gpus_per_node": args.gpus,
            "trainer.nnodes": args.nodes,
            "trainer.default_local_dir": str(output / "checkpoints"),
            "trainer.total_epochs": args.epochs,
            "trainer.total_training_steps": args.steps,
            "trainer.test_freq": args.eval_every,
            "trainer.save_freq": args.save_every,
            "trainer.save_vector": vector,
            "trainer.save_vector_dir": str(output / "vectors"),
            "trainer.validation_data_dir": str(output / "validation"),
        }
        if "qwen3" in model.lower():
            opts["+data.apply_chat_template_kwargs.enable_thinking"] = False
        if not rl:
            opts["+actor_rollout_ref.ref.model.path"] = teacher
            opts["algorithm.rollout_correction.rollout_is"] = "token" if args.stage == "opd" else None
            opts["algorithm.rollout_correction.rollout_is_threshold"] = 5.0
        elif args.rl_algorithm == "dapo":
            # Use the DAPO trainer so dynamic group filtering is actually applied.
            opts.update(
                {
                    "actor_rollout_ref.actor.clip_ratio_low": 0.2,
                    "actor_rollout_ref.actor.clip_ratio_high": 0.28,
                    "algorithm.filter_groups.enable": True,
                    "algorithm.filter_groups.metric": "seq_reward",
                    "algorithm.filter_groups.max_num_gen_batches": 10,
                    "reward_model.reward_manager": "dapo",
                }
            )
        if vector:
            first, last = (int(x) for x in args.layers.split(":"))
            if first < 0 or last < first:
                raise ValueError("--layers must be a nonnegative inclusive start:end range")
            opts.update(
                {
                    "actor_rollout_ref.model.trainable_token_vector_mode": "multi"
                    if args.method == "sequential"
                    else "single",
                    "actor_rollout_ref.model.trainable_token_vector_num": args.vectors
                    if args.method == "sequential"
                    else 1,
                    "actor_rollout_ref.model.trainable_token_vector_layer_start": first,
                    "actor_rollout_ref.model.trainable_token_vector_layer_end": last,
                    "actor_rollout_ref.model.trainable_token_vector_scale": 0.1,
                    "actor_rollout_ref.model.trainable_token_vector_learnable_alpha": args.method == "sequential",
                    "actor_rollout_ref.model.trainable_token_vector_force_all_tokens": True,
                    "actor_rollout_ref.model.trainable_token_vector_curriculum": "sequential_orthogonal"
                    if args.method == "sequential"
                    else "none",
                    "actor_rollout_ref.model.trainable_token_vector_seq_raw_steps": args.stage_steps,
                    "actor_rollout_ref.model.trainable_token_vector_seq_max_iters": args.stage_steps,
                    "actor_rollout_ref.model.trainable_token_vector_gated": args.method == "gated",
                    "actor_rollout_ref.model.trainable_token_vector_gate_rank": args.gate_rank
                    if args.method == "gated"
                    else 0,
                }
            )
        if args.stage == "offpolicy":
            if not args.offline_data:
                raise ValueError("offpolicy requires --offline-data from generate + merge_rollouts.py")
            opts["data.train_files"] = str(Path(args.offline_data).resolve())
            opts.update(
                {
                    "actor_rollout_ref.rollout.offline_teacher_rollout": True,
                    "actor_rollout_ref.rollout.offline_teacher_data_path": str(Path(args.offline_data).resolve()),
                    "actor_rollout_ref.rollout.offline_teacher_response_key": "response",
                }
            )
        if args.stage == "alpha":
            if vector:
                raise ValueError("alpha supports --method full or lora")
            opts["+actor_rollout_ref.model.base_model_path"] = model
            opts["actor_rollout_ref.actor.alpha_stabler.enabled"] = True
            opts["actor_rollout_ref.actor.alpha_stabler.control"] = not args.monitor_only
            opts["actor_rollout_ref.actor.use_torch_compile"] = False
            opts["actor_rollout_ref.ref.use_torch_compile"] = False
            opts["actor_rollout_ref.rollout.enforce_eager"] = True
            opts["ray_kwargs.ray_init.num_cpus"] = args.ray_cpus
        if args.domain in ("science", "instruction"):
            scorer = "sciknoweval_verify" if args.domain == "science" else "ifevalg_verify"
            opts["custom_reward_function.path"] = str(ROOT / f"verl/utils/reward_score/{scorer}.py")
        if args.domain == "code":
            sandbox_url = args.sandbox_url or os.environ.get("SANDBOX_FUSION_URL")
            if sandbox_url:
                opts["reward_model.sandbox_fusion.url"] = sandbox_url
        module = "verl.trainer.main_ppo"
        if rl and args.rl_algorithm == "dapo":
            module = "recipe.dapo.main_dapo"
    # Explicit Hydra overrides are last, so users can reproduce individual runs.
    return [sys.executable, "-m", module] + [f"{k}={value(v)}" for k, v in opts.items()] + extra


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["rl", "opd", "generate", "offpolicy", "alpha"])
    p.add_argument("domain", choices=list(TEACHERS))
    p.add_argument("--method", choices=["full", "lora", "vector", "sequential", "gated"], default="full")
    p.add_argument("--model")
    p.add_argument("--teacher")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--train-file")
    p.add_argument("--val-files", nargs="+")
    p.add_argument("--offline-data")
    p.add_argument("--output-dir", default=str(ROOT / "outputs"))
    p.add_argument("--name")
    p.add_argument("--rl-algorithm", choices=["grpo", "dapo"], default="grpo")
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--mini-batch", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument("--max-prompt", type=int)
    p.add_argument("--max-response", type=int)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--tp", type=int)
    p.add_argument("--gpu-memory", type=float, default=0.6)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-max-samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--layers", default="5:20", help="Zero-based inclusive start:end")
    p.add_argument("--vectors", type=int, default=8)
    p.add_argument("--stage-steps", type=int, default=100)
    p.add_argument("--gate-rank", type=int, default=1)
    p.add_argument("--monitor-only", action="store_true")
    p.add_argument("--sandbox-url")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config-only", action="store_true", help="Compose/resolve Hydra config without launching training")
    p.add_argument("--check", action="store_true", help="Check training dependencies, model and data without training")
    p.add_argument("--no-download", action="store_true", help="Require model/data to be available locally")
    p.add_argument("--ray-cpus", type=int, default=32, help="CPU budget for the local Ray runtime")
    return p


def compose_config(command):
    from hydra import compose, initialize_config_dir

    dapo = command[2] == "recipe.dapo.main_dapo"
    with initialize_config_dir(
        config_dir=str(ROOT / ("recipe/dapo/config" if dapo else "verl/trainer/config")), version_base=None
    ):
        return compose(
            config_name="dapo_trainer"
            if dapo
            else "generation"
            if command[2].endswith("main_generation")
            else "ppo_trainer",
            overrides=command[3:],
        )


def prepare_alpha(args, command):
    """Resolve the effective configuration, then validate/download only its inputs."""
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from omegaconf import OmegaConf

    config = compose_config(command)
    OmegaConf.resolve(config)
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("Alpha-Stabler requires FSDP")
    if config.actor_rollout_ref.rollout.mode != "sync":
        raise ValueError("Alpha-Stabler requires synchronous rollout")
    if config.actor_rollout_ref.actor.ulysses_sequence_parallel_size != 1:
        raise ValueError("Alpha-Stabler currently requires Ulysses sequence parallelism 1")
    gpu_count = torch.cuda.device_count()
    if gpu_count < config.trainer.n_gpus_per_node:
        raise RuntimeError(
            f"Requested {config.trainer.n_gpus_per_node} GPUs/node, but {gpu_count} CUDA GPUs are visible"
        )
    for module in ("ray", "vllm", "flash_attn", "verl.trainer.main_ppo", "verl.workers.fsdp_workers"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import {module}: {exc}. Install the training dependencies in this Python environment."
            ) from exc
    if args.domain == "instruction":
        importlib.import_module("verl.utils.reward_score.ifevalg_verify")
        importlib.import_module("verl.utils.reward_score.ifbench.instructions")
    elif args.domain == "math":
        importlib.import_module("math_verify")

    pin = json.loads((ROOT / "configs/datasets.json").read_text())[args.domain]
    data_dir = Path(args.data_dir).resolve() / args.domain
    files = []
    for configured in (config.data.train_files, config.data.val_files):
        files.extend([configured] if isinstance(configured, str) else list(configured))
    for item in files:
        path = Path(item).expanduser().resolve()
        if not path.is_file():
            if args.no_download or path.parent != data_dir:
                raise FileNotFoundError(f"Dataset not found: {path}")
            print(f"Downloading {pin['repo_id']}/{path.name}", flush=True)
            hf_hub_download(
                pin["repo_id"], path.name, repo_type="dataset", revision=pin["revision"], local_dir=data_dir
            )
        import pyarrow.parquet as pq

        dataset = pq.ParquetFile(path)
        required = {"prompt", "data_source", "reward_model"}
        if not required.issubset(dataset.schema_arrow.names) or dataset.metadata.num_rows == 0:
            raise ValueError(f"Invalid or empty training dataset: {path}")
        print(f"Ready: {path.name} ({dataset.metadata.num_rows:,} rows)", flush=True)

    model_id = config.actor_rollout_ref.model.path
    base_id = config.actor_rollout_ref.model.get("base_model_path")
    if model_id != base_id:
        raise ValueError("Alpha-Stabler model.path and model.base_model_path must identify the same initial model")
    local_model = Path(model_id).expanduser()
    if not local_model.is_dir():
        if model_id.startswith(("/", ".", "~")):
            raise FileNotFoundError(f"Model directory not found: {local_model}")
        print(f"Resolving Hugging Face model: {model_id}", flush=True)
        local_model = Path(
            snapshot_download(
                model_id,
                local_files_only=args.no_download,
                allow_patterns=[
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "config.json",
                    "generation_config.json",
                    "tokenizer*",
                    "vocab.json",
                    "merges.txt",
                    "added_tokens.json",
                    "special_tokens_map.json",
                    "chat_template*",
                ],
            )
        )
    from transformers import AutoConfig, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(local_model, local_files_only=True)
    AutoTokenizer.from_pretrained(local_model, local_files_only=True)
    if not list(local_model.glob("*.safetensors")) and not (local_model / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Missing model weights in {local_model}")
    layers = config.actor_rollout_ref.actor.alpha_stabler.layers
    if layers and any(index < 0 or index >= hf_config.num_hidden_layers for index in layers):
        raise ValueError("Alpha-Stabler layer selection exceeds model depth")
    print(f"Ready: {hf_config.model_type}, {hf_config.num_hidden_layers} layers, {gpu_count} visible GPUs", flush=True)
    # Replace paths in place so user overrides do not produce duplicate Hydra keys.
    replacements = {
        "actor_rollout_ref.model.path": str(local_model.resolve()),
        "actor_rollout_ref.model.base_model_path": str(local_model.resolve()),
    }
    return [
        f"{item.split('=', 1)[0]}={value(replacements[item.split('=', 1)[0].lstrip('+')])}"
        if "=" in item and item.split("=", 1)[0].lstrip("+") in replacements
        else item
        for item in command
    ]


def main():
    args, extra = parser().parse_known_args()
    if any("=" not in item for item in extra):
        raise SystemExit(f"Unknown arguments (Hydra overrides must contain '='): {extra}")
    command = build_command(args, extra)
    if args.config_only:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(compose_config(command), resolve=True))
    elif args.dry_run:
        print(shlex.join(command), flush=True)
    elif not args.dry_run:
        if args.stage == "alpha":
            # The synchronous external-launcher path uses the vLLM V0 engine.
            os.environ.setdefault("VLLM_USE_V1", "0")
            command = prepare_alpha(args, command)
        elif args.check:
            raise ValueError("--check is currently supported by the alpha stage")
        if args.check:
            print("Alpha-Stabler startup check passed.")
            return
        run_dir = Path(args.output_dir).resolve() / (args.name or f"{args.domain}-{args.stage}-{args.method}")
        run_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("VERL_FILE_LOGGER_PATH", str(run_dir / "metrics.jsonl"))
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("PYTHONUNBUFFERED", "1")
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
