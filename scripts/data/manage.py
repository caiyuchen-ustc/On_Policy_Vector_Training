#!/usr/bin/env python3
"""Prepare, audit, publish, and download the four original experiment datasets.

No credentials are saved. Publication reads HF_TOKEN or a non-echoing prompt.
Parquet files are copied byte-for-byte: no schema or data transformations.
"""

import argparse
import getpass
import hashlib
import json
import os
import shutil
from pathlib import Path

DOMAINS = {
    "math": {
        "repo": "caiyuchen/OPV-Math",
        "sources": ["DeepMath-Team/DeepMath-103K", "BytedTsinghua-SIA/DAPO-Math-17k", "HuggingFaceH4/aime_2024"],
        "files": {
            "train": "G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
            "teacher_train": "DAPO/level6_prompt.parquet",
            "validation": "G-OPD-Training-Data/AIME2024/test.parquet",
            "aime2025": "G-OPD-Training-Data/AIME2025/test.parquet",
        },
        "note": (
            "train is filtered DeepMath (distillation); teacher_train is the original DAPO teacher corpus. "
            "validation is AIME2024. These corpora are not interchangeable."
        ),
    },
    "science": {
        "repo": "caiyuchen/OPV-Science",
        "sources": ["hicai-zju/SciKnowEval"],
        "files": {
            "train": "sciknoweval/sciknoweval_train.parquet",
            "validation": "sciknoweval/sciknoweval_validation.parquet",
        },
        "note": (
            "Original four-domain split (physics, chemistry, biology, material). "
            "Original system/user prompts and answer-tag instructions are preserved."
        ),
    },
    "code": {
        "repo": "caiyuchen/OPV-Code",
        "sources": ["PRIME-RL/Eurus-2-RL-Data", "livecodebench/code_generation_lite"],
        "files": {
            "train": "G-OPD-Training-Data/Eurus/code_train.parquet",
            "validation": "G-OPD-Training-Data/Eurus/livecodebench_v5.parquet",
            "eurus_validation": "G-OPD-Training-Data/Eurus/code_validation.parquet",
        },
        "note": (
            "validation is the full local LiveCodeBench v5 export (880 problems), not the 100-row smoke subset. "
            "The original converter caps tests at 30 and their combined bytes; this is an execution-budget variant, "
            "not unrestricted official LCB evaluation."
        ),
    },
    "instruction": {
        "repo": "caiyuchen/OPV-Instruction",
        "sources": ["nvidia/Nemotron-3-Nano-RL-Training-Blend", "allenai/IFBench_test"],
        "files": {"train": "ifevalg/ifevalg_train.parquet", "validation": "ifevalg/ifevalg_validation.parquet"},
        "note": (
            "Original Nemotron instruction-following subset and independent AllenAI IFBench test set. "
            "This is NOT an 80/20 IFEval split; the manuscript description differs "
            "from the available experiment artifacts."
        ),
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    import pyarrow.parquet as pq

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for name in args.domains:
        spec = DOMAINS[name]
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        report = {
            "domain": name,
            "repo_id": spec["repo"],
            "sources": spec["sources"],
            "note": spec["note"],
            "splits": {},
        }
        prompt_sets = {}
        files = {k: Path(args.source) / v for k, v in spec["files"].items()}
        for split, source in files.items():
            target = dest / f"{split}.parquet"
            seen = set()
            rows = 0
            domains = {}
            if source.resolve() == target.resolve():
                raise ValueError("Staging destination must differ from the original data directory")
            shutil.copy2(source, target)
            for batch in pq.ParquetFile(source).iter_batches(batch_size=128, columns=["prompt", "extra_info"]):
                for row in batch.to_pylist():
                    key = json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)
                    seen.add(hashlib.sha256(key.encode()).hexdigest())
                    domain = (row.get("extra_info") or {}).get("domain")
                    if domain:
                        domains[domain] = domains.get(domain, 0) + 1
                rows += batch.num_rows
            prompt_sets[split] = seen
            report["splits"][split] = {
                "rows": rows,
                "unique_prompts": len(seen),
                "sha256": sha256(target),
                "source_sha256": sha256(source),
                "source_file": source.name,
                "domains": domains,
            }
            if report["splits"][split]["sha256"] != report["splits"][split]["source_sha256"]:
                raise RuntimeError(f"Copy verification failed: {source}")
            print(f"{name}/{split}: {rows} rows, {len(seen)} prompts", flush=True)
        report["train_validation_exact_prompt_overlap"] = len(prompt_sets["train"] & prompt_sets["validation"])
        if report["train_validation_exact_prompt_overlap"]:
            print(
                f"WARNING: {name} has {report['train_validation_exact_prompt_overlap']} "
                "original train/validation prompt overlap(s); preserved and disclosed",
                flush=True,
            )
        (dest / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        # Original train/evaluation Arrow schemas may differ. Give each file its
        # own Hub configuration so the viewer never attempts to coerce schemas.
        config_lines = "\n".join(
            f"- config_name: {split}\n  data_files:\n  - split: {split}\n    path: {split}.parquet" for split in files
        )
        counts = "\n".join(f"| {s} | {v['rows']:,} |" for s, v in report["splits"].items())
        links = "\n".join(f"- https://huggingface.co/datasets/{s}" for s in spec["sources"])
        card = f'''---
language:
- en
task_categories:
- text-generation
tags:
- reinforcement-learning
- verl
- opv
configs:
{config_lines}
---
# OPV {name.title()}: original experiment data

Prepared for *Learning to Steer, Steering to See*. {spec["note"]}

| Split | Rows |
| --- | ---: |
{counts}

## Provenance and terms

The data is derived from the following sources; their original terms and attribution obligations continue to apply.
No new blanket license is asserted over the collection. Source-file and uploaded-file hashes are in `manifest.json`.

{links}

## Format and evaluation

These are byte-for-byte copies of the original Parquet files. All columns, Arrow schemas, nested verifier metadata,
row order, prompts and answers are unchanged. Use the existing verl reward functions directly.
Each file has a separate Hub configuration because original train/evaluation schemas differ.

Use the full evaluation split and four independently sampled responses per prompt; report **mean@4**, not pass@4.
No training responses or model weights are included. Exact normalized-chat train/evaluation overlap:
**{report["train_validation_exact_prompt_overlap"]} prompts**. Original splits are preserved, including any
disclosed overlap, to match existing teacher provenance. This is not a semantic contamination audit.

```python
from datasets import load_dataset
train = load_dataset("{spec["repo"]}", "train", split="train")
validation = load_dataset("{spec["repo"]}", "validation", split="validation")
```

The available artifacts differ from several manuscript descriptions. The accompanying repository records these
differences instead of relabeling datasets or inventing results.
'''
        (dest / "README.md").write_text(card)


def publish(args):
    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token (not saved): ")
    api = HfApi(token=token)
    if api.whoami()["name"] != "caiyuchen":
        raise RuntimeError("Token identity does not match requested caiyuchen namespace")
    receipts_path = Path(args.output) / "upload_receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    for name in args.domains:
        folder = Path(args.output) / name
        report = json.loads((folder / "manifest.json").read_text())
        for split, record in report["splits"].items():
            if sha256(folder / f"{split}.parquet") != record["sha256"]:
                raise RuntimeError(f"Changed artifact: {name}/{split}")
        repo = DOMAINS[name]["repo"]
        api.create_repo(repo, repo_type="dataset", private=args.private, exist_ok=True)
        paths = [folder / f"{s}.parquet" for s in report["splits"]] + [folder / "README.md", folder / "manifest.json"]
        commit = api.create_commit(
            repo,
            repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in paths],
            commit_message="Publish audited OPV experiment splits and provenance",
        )
        receipts[name] = {"repo_id": repo, "revision": commit.oid, "url": f"https://huggingface.co/datasets/{repo}"}
        (Path(args.output) / "upload_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps(receipts[name]), flush=True)


def download(args):
    from huggingface_hub import snapshot_download

    pins = json.loads(Path(args.pins).read_text()) if args.pins else {}
    for name in args.domains:
        snapshot_download(
            DOMAINS[name]["repo"],
            repo_type="dataset",
            revision=pins.get(name, {}).get("revision", "main"),
            local_dir=str(Path(args.output) / name),
            allow_patterns=["*.parquet", "manifest.json", "README.md"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "publish", "download"])
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS), default=list(DOMAINS))
    parser.add_argument("--source", default="../data")
    parser.add_argument("--output", default="data")
    parser.add_argument(
        "--pins",
        default=str(Path(__file__).resolve().parents[2] / "configs/datasets.json"),
        help="JSON mapping domains to immutable HF revisions",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "publish": publish, "download": download}[args.action](args)


if __name__ == "__main__":
    main()
