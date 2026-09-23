"""Portable roots for legacy analyses that expect local models/vector artifacts."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def project_path(suffix=""):
    return str(REPO / suffix)


def model_path(suffix=""):
    # Some analyses read safetensor shards directly, so this is a local directory.
    return str(Path(os.environ.get("OPV_MODEL_ROOT", REPO / "models")) / suffix)


def artifact_path(suffix=""):
    return str(Path(os.environ.get("OPV_ARTIFACT_ROOT", REPO / "artifacts")) / suffix)


def data_path(suffix=""):
    root = Path(os.environ.get("OPV_DATA_ROOT", REPO / "data"))
    names = {
        "G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet": "math/train.parquet",
        "G-OPD-Training-Data/AIME2024/test.parquet": "math/validation.parquet",
        "G-OPD-Training-Data/AIME2025/test.parquet": "math/aime2025.parquet",
        "DAPO/level6_prompt.parquet": "math/teacher_train.parquet",
        "sciknoweval/sciknoweval_train.parquet": "science/train.parquet",
        "sciknoweval/sciknoweval_validation.parquet": "science/validation.parquet",
        "ifevalg/ifevalg_train.parquet": "instruction/train.parquet",
        "ifevalg/ifevalg_validation.parquet": "instruction/validation.parquet",
        "G-OPD-Training-Data/Eurus/code_train.parquet": "code/train.parquet",
        "G-OPD-Training-Data/Eurus/code_validation.parquet": "code/eurus_validation.parquet",
        "G-OPD-Training-Data/Eurus/livecodebench_v5.parquet": "code/validation.parquet",
    }
    return str(root / names.get(suffix, suffix))
