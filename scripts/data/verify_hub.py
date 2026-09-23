#!/usr/bin/env python3
"""Verify every published Parquet against both its source and Hub object hash."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def digest(path, git=False):
    result = hashlib.sha1() if git else hashlib.sha256()
    if git:
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pins", default="configs/datasets.json")
    args = parser.parse_args()
    pins = json.loads(Path(args.pins).read_text())
    api = HfApi()
    for domain, pin in pins.items():
        folder = Path(args.data_dir) / domain
        manifest = json.loads((folder / "manifest.json").read_text())
        info = api.dataset_info(pin["repo_id"], revision=pin["revision"], files_metadata=True)
        files = {entry.rfilename: entry for entry in info.siblings}
        for split, record in manifest["splits"].items():
            name = split + ".parquet"
            path = folder / name
            local_sha = digest(path)
            assert local_sha == record["sha256"] == record["source_sha256"], f"Local/source mismatch: {domain}/{split}"
            entry = files[name]
            if entry.lfs:
                assert entry.lfs.sha256 == local_sha, f"Hub LFS hash mismatch: {domain}/{split}"
            else:
                assert entry.blob_id == digest(path, git=True), f"Hub Git hash mismatch: {domain}/{split}"
            print(f"PASS {pin['repo_id']}/{name}: original bytes verified", flush=True)


if __name__ == "__main__":
    main()
