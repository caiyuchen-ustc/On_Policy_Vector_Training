#!/usr/bin/env python3
"""Plot actual verl file-logger metrics, without generating or filling measurements."""

import argparse
import json
from pathlib import Path


def read_metrics(path):
    records = {}
    for index, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "step" not in record or not isinstance(record.get("data"), dict):
            raise ValueError(f"{path}:{index}: expected verl {{step, data}} record")
        records.setdefault(record["step"], {}).update(record["data"])
    return sorted(records.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--keys", nargs="+", default=["alpha_stabler/psi", "alpha_stabler/active_layers"])
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--list-keys", action="store_true")
    parser.add_argument("--output", default="analysis/figures/metrics.png")
    args = parser.parse_args()
    runs = [read_metrics(path) for path in args.logs]
    if args.list_keys:
        print("\n".join(sorted({key for run in runs for _, data in run for key in data})))
        return
    labels = args.labels or [Path(path).parent.name for path in args.logs]
    if len(labels) != len(runs):
        parser.error("--labels must match the number of logs")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.keys), 1, figsize=(8, 3 * len(args.keys)), squeeze=False)
    for key, ax in zip(args.keys, axes.flat, strict=True):
        for run, label in zip(runs, labels, strict=True):
            points = [(step, data[key]) for step, data in run if key in data]
            if not points:
                raise ValueError(f"No measurements for {key!r} in {label!r}; use --list-keys")
            x, y = zip(*points, strict=True)
            ax.plot(x, y, label=label, linewidth=1.3)
        ax.set(xlabel="Training step", ylabel=key)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
