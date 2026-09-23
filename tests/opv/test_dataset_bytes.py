import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq


def test_packaging_preserves_original_parquet_bytes_and_nested_schema(tmp_path):
    path = Path(__file__).parents[2] / "scripts/data/manage.py"
    spec = importlib.util.spec_from_file_location("opv_data", path)
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    data.DOMAINS = {
        "example": {
            "repo": "caiyuchen/example",
            "sources": [],
            "note": "test",
            "files": {"train": "train.parquet", "validation": "validation.parquet"},
        }
    }
    for split in ("train", "validation"):
        table = pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": split}],
                    "reward_model": {"ground_truth": "A", "style": "rule"},
                    "extra_info": {
                        "kwargs": [{"nested": 3, "nullable": None}],
                        "index": 7 if split == "train" else "id7",
                    },
                    "original_extra_column": [1, 2, 3],
                }
            ]
        )
        pq.write_table(table, tmp_path / f"{split}.parquet")
    data.prepare(SimpleNamespace(source=tmp_path, output=tmp_path / "staging", domains=["example"]))
    for split in ("train", "validation"):
        assert (tmp_path / f"{split}.parquet").read_bytes() == (
            tmp_path / "staging/example" / f"{split}.parquet"
        ).read_bytes()
    manifest = json.loads((tmp_path / "staging/example/manifest.json").read_text())
    assert manifest["train_validation_exact_prompt_overlap"] == 0
