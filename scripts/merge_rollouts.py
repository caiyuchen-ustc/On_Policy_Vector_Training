#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Merge teacher rollout dumps (one parquet per round, produced by
verl.trainer.main_generation) into a single SFT-ready parquet.

Each round parquet is the original prompt dataset with an added `responses`
column, where `responses[i]` is a list of `n_samples` teacher generations for
row i. This script explodes that into one (prompt, response) pair per generation
so the result is directly consumable by verl.utils.dataset.sft_dataset.SFTDataset
(prompt_key=prompt, response_key=response).

Usage:
  python merge_rollouts_to_sft.py \
      --inputs round_0.parquet round_1.parquet ... \
      --output teacher_sft_all.parquet \
      [--prompt_key prompt] [--response_key response] \
      [--min_response_chars 1]
"""

import argparse
import glob
import os


def parse_args():
    ap = argparse.ArgumentParser(description="Merge teacher rollout rounds into an SFT parquet.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Round parquet files (supports globs, e.g. '.../round_*.parquet').",
    )
    ap.add_argument("--output", required=True, help="Output SFT parquet path.")
    ap.add_argument("--prompt_key", default="prompt", help="Prompt column name (chat list).")
    ap.add_argument("--gen_key", default="responses", help="Generated column produced by main_generation.")
    ap.add_argument("--response_key", default="response", help="Output response column name for SFTDataset.")
    ap.add_argument(
        "--min_response_chars",
        type=int,
        default=1,
        help="Drop (prompt, response) pairs whose response has fewer than this many characters.",
    )
    return ap.parse_args()


def _resolve_inputs(patterns):
    files = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if matched:
            files.extend(matched)
        elif os.path.exists(pat):
            files.append(pat)
        else:
            raise FileNotFoundError(f"No parquet matched: {pat}")
    # de-dup while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def main():
    args = parse_args()
    files = _resolve_inputs(args.inputs)
    if os.path.abspath(args.output) in {os.path.abspath(path) for path in files}:
        raise ValueError("Output must not overwrite an input rollout file")
    if os.path.exists(args.output):
        raise FileExistsError(f"Output already exists: {args.output}; choose a new output path")
    print(f"Merging {len(files)} round file(s):")
    for f in files:
        print(f"  - {f}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    n_written = 0
    n_dropped = 0
    writer = None
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for f in files:
        pf = pq.ParquetFile(f)
        # (prompt may appear as a nested struct whose fields are flattened in schema.names; the
        #  pandas frame from a record batch still exposes it as a single column, so we check per-batch.)
        for batch in pf.iter_batches(batch_size=512):
            df = batch.to_pandas()
            if args.prompt_key not in df.columns or args.gen_key not in df.columns:
                raise KeyError(
                    f"{f} missing columns; has {list(df.columns)}, need '{args.prompt_key}' and '{args.gen_key}'."
                )
            passthrough_cols = [c for c in df.columns if c != args.gen_key]
            rows = []
            for _, row in df.iterrows():
                gens = row[args.gen_key]
                if isinstance(gens, str):
                    gens = [gens]
                for g in gens:
                    g = "" if g is None else str(g)
                    if len(g.strip()) < args.min_response_chars:
                        n_dropped += 1
                        continue
                    new_row = {c: row[c] for c in passthrough_cols}
                    new_row[args.response_key] = g
                    rows.append(new_row)
            if not rows:
                continue
            out_tbl = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(args.output, out_tbl.schema)
            else:
                out_tbl = out_tbl.cast(writer.schema, safe=False)
            writer.write_table(out_tbl)
            n_written += len(rows)

    if writer is not None:
        writer.close()
    else:
        raise ValueError("No nonempty teacher responses found; no distillation corpus was written")
    print(
        f"Wrote {n_written} (prompt, response) pairs to {args.output} (dropped {n_dropped} empty/too-short responses)."
    )


if __name__ == "__main__":
    main()
