import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def test_metric_logger_serializes_tensor_numpy_and_resume(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "verl/utils/tracking.py"
    spec = importlib.util.spec_from_file_location("opv_tracking", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(output))
    logger = module.FileLogger("test", "run")
    logger.log({"tensor": torch.tensor(0.25, requires_grad=True), "array": np.array([1.0, 2.0])}, np.int64(1))
    logger.finish()
    logger = module.FileLogger("test", "run")
    logger.log({"score": np.float32(0.5)}, 2)
    logger.finish()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == [{"step": 1, "data": {"tensor": 0.25, "array": [1.0, 2.0]}}, {"step": 2, "data": {"score": 0.5}}]
