#!/usr/bin/env python3
"""Offline validation of portable launchers, shell syntax, and credential hygiene."""

import ast
import importlib.util
import itertools
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location("opv_train", ROOT / "scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    checked = 0
    for stage, domain, method in itertools.product(
        ["rl", "opd", "offpolicy", "generate", "alpha"],
        train.TEACHERS,
        ["full", "lora", "vector", "sequential", "gated"],
    ):
        if (stage == "alpha" and method not in ("full", "lora")) or (stage == "generate" and method != "full"):
            continue
        args = train.parser().parse_args([stage, domain, "--method", method, "--offline-data", "/tmp/teacher.parquet"])
        command = train.build_command(args, [])
        with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
            config = compose(config_name="generation" if stage == "generate" else "ppo_trainer", overrides=command[3:])
            OmegaConf.resolve(config)
        checked += 1
    for domain in train.TEACHERS:
        args = train.parser().parse_args(["rl", domain, "--rl-algorithm", "dapo"])
        command = train.build_command(args, [])
        with initialize_config_dir(config_dir=str(ROOT / "recipe/dapo/config"), version_base=None):
            config = compose(config_name="dapo_trainer", overrides=command[3:])
            OmegaConf.resolve(config)
        checked += 1
    paths = [p for base in ("scripts", "examples", "recipe") for p in (ROOT / base).rglob("*.sh")]
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    for directory in ("scripts", "verl", "tests/opv"):
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
    secret_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}|WANDB_API_KEY\s*=\s*['\"]?[a-f0-9]{30,}")
    for directory in ("scripts", "examples", "recipe"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in (".py", ".sh", ".yaml", ".yml", ".md") and secret_pattern.search(path.read_text()):
                raise RuntimeError(f"Embedded credential found in {path.relative_to(ROOT)}")
    print(f"PASS: {checked} training configurations; {len(paths)} shell scripts; Python syntax; credential scan")


if __name__ == "__main__":
    main()
