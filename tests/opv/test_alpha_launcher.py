import importlib.util
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("opv_launcher", ROOT / "scripts/train.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_shell_defaults_to_science_from_another_directory(tmp_path):
    env = dict(os.environ, PYTHON=os.sys.executable)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/alpha_stabler/run.sh"), "--config-only"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "experiment_name: science-alpha-full" in result.stdout
    assert "enabled: true" in result.stdout
    assert "use_torch_compile: false" in result.stdout
    assert "enforce_eager: true" in result.stdout


def test_alpha_cli_overrides_are_composed_for_preflight():
    args = launcher.parser().parse_args(["alpha", "science", "--gpus", "2", "--tp", "2"])
    config = launcher.compose_config(launcher.build_command(args, ["data.train_files=/tmp/custom.parquet"]))
    assert config.data.train_files == "/tmp/custom.parquet"
    assert config.actor_rollout_ref.model.base_model_path == config.actor_rollout_ref.model.path
    assert config.actor_rollout_ref.actor.alpha_stabler.enabled
    assert config.ray_kwargs.ray_init.num_cpus == 32


def test_task_environment_selects_another_default(tmp_path):
    env = dict(os.environ, PYTHON=os.sys.executable, TASK="instruction")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/alpha_stabler/run.sh"), "--dry-run"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "instruction-alpha-full" in result.stdout
    assert "DeepSeek-R1-Distill-Qwen-7B" in result.stdout
