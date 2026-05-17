#!/usr/bin/env python3
"""Build a best-candidate portfolio from arbitrary per-instance CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: str, source: str) -> dict[str, dict]:
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = str(row.get("index", len(rows)))
            rows[key] = {
                "index": key,
                "file": row.get("file", ""),
                "instance_id": row.get("instance_id", ""),
                "source": source,
                "objective": float(row["objective_value"]),
                "solved": int(row.get("solved", 0)),
                "route": row.get("route", "[]"),
            }
    return rows


def trimmed_mean(values: np.ndarray, frac: float = 0.05) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    values = np.sort(values)
    k = int(math.floor(values.size * frac))
    if 2 * k >= values.size:
        return float(np.mean(values))
    return float(np.mean(values[k : values.size - k]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Reference CSV for delta calculation.")
    parser.add_argument("--candidate", action="append", required=True, help="name=csv_path")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    base = read_csv(args.base, "base")
    parts = []
    for spec in args.candidate:
        if "=" not in spec:
            raise ValueError(f"candidate must be name=path, got: {spec}")
        name, path = spec.split("=", 1)
        parts.append(read_csv(path, name))
    keys = sorted(set(base).intersection(*[set(p) for p in parts]), key=lambda x: int(x))
    output = {}
    source_counts: dict[str, int] = {}
    for key in keys:
        best = min((p[key] for p in parts), key=lambda row: row["objective"])
        output[key] = dict(best)
        source_counts[best["source"]] = source_counts.get(best["source"], 0) + 1

    deltas = np.asarray([output[k]["objective"] - base[k]["objective"] for k in keys], dtype=np.float64)
    finite = np.isfinite(deltas)
    finite_deltas = deltas[finite]

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[keys[0]].keys()))
        writer.writeheader()
        for key in keys:
            writer.writerow(output[key])

    summary = {
        "instances": int(len(keys)),
        "finite": int(finite_deltas.size),
        "portfolio_avg_obj": float(np.mean([output[k]["objective"] for k in keys])),
        "base_avg_obj": float(np.mean([base[k]["objective"] for k in keys])),
        "mean_delta": float(np.mean(finite_deltas)) if finite_deltas.size else float("nan"),
        "median_delta": float(np.median(finite_deltas)) if finite_deltas.size else float("nan"),
        "trimmed_mean_delta": trimmed_mean(finite_deltas),
        "wins": int(np.sum(finite_deltas < -1e-9)),
        "losses": int(np.sum(finite_deltas > 1e-9)),
        "ties": int(np.sum(np.abs(finite_deltas) <= 1e-9)),
        "source_counts": source_counts,
        "base": args.base,
        "candidates": args.candidate,
        "output_csv": args.output_csv,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[FlexiblePortfolio] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
