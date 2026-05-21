#!/usr/bin/env python3
"""Summarize candidate-level portfolios from policy_eval CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path, source):
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


def finite_array(values):
    return np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)


def trimmed_mean(values, frac=0.05):
    arr = finite_array(values)
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr)
    k = int(math.floor(arr.size * frac))
    if 2 * k >= arr.size:
        return float(np.mean(arr))
    return float(np.mean(arr[k : arr.size - k]))


def compare(name, cand, base):
    keys = sorted(set(base.keys()) & set(cand.keys()), key=lambda x: int(x))
    deltas = np.asarray([cand[k]["objective"] - base[k]["objective"] for k in keys], dtype=np.float64)
    finite = np.isfinite(deltas)
    deltas_f = deltas[finite]
    if deltas_f.size == 0:
        return {}
    return {
        "name": name,
        "instances": int(len(keys)),
        "finite": int(deltas_f.size),
        "avg_obj": float(np.mean([cand[k]["objective"] for k in keys if np.isfinite(cand[k]["objective"])])),
        "base_avg_obj": float(np.mean([base[k]["objective"] for k in keys if np.isfinite(base[k]["objective"])])),
        "mean_delta": float(np.mean(deltas_f)),
        "median_delta": float(np.median(deltas_f)),
        "trimmed_mean_delta": trimmed_mean(deltas_f),
        "wins": int(np.sum(deltas_f < -1e-9)),
        "losses": int(np.sum(deltas_f > 1e-9)),
        "ties": int(np.sum(np.abs(deltas_f) <= 1e-9)),
        "win_rate": float(np.mean(deltas_f < -1e-9)),
        "loss_rate": float(np.mean(deltas_f > 1e-9)),
    }


def make_portfolio(parts, name):
    keys = sorted(set.intersection(*[set(p.keys()) for p in parts]), key=lambda x: int(x))
    out = {}
    source_counts = {}
    for key in keys:
        best = min((p[key] for p in parts), key=lambda row: row["objective"])
        out[key] = dict(best)
        out[key]["source"] = name + ":" + best["source"]
        source_counts[best["source"]] = source_counts.get(best["source"], 0) + 1
    return out, source_counts


def write_rows(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(rows.keys(), key=lambda x: int(x))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[keys[0]].keys()))
        writer.writeheader()
        for key in keys:
            writer.writerow(rows[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base50", required=True)
    parser.add_argument("--base40", required=True)
    parser.add_argument("--base32", required=True)
    parser.add_argument("--temp10", required=True)
    parser.add_argument("--temp8", required=True)
    parser.add_argument("--off10", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base50 = read_csv(args.base50, "base50")
    base40 = read_csv(args.base40, "base40")
    base32 = read_csv(args.base32, "base32")
    temp10 = read_csv(args.temp10, "temp10")
    temp8 = read_csv(args.temp8, "temp8")
    off10 = read_csv(args.off10, "off10")

    portfolios = {
        "base40_temp10": make_portfolio([base40, temp10], "base40_temp10"),
        "base40_off10": make_portfolio([base40, off10], "base40_off10"),
        "base32_temp8_off10": make_portfolio([base32, temp8, off10], "base32_temp8_off10"),
    }
    comparisons = [
        compare("base40", base40, base50),
        compare("base32", base32, base50),
        compare("temp10_only", temp10, base50),
        compare("off10_only", off10, base50),
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_counts = {}
    for name, (rows, counts) in portfolios.items():
        comparisons.append(compare(name, rows, base50))
        source_counts[name] = counts
        write_rows(output_dir / f"{name}.csv", rows)

    summary = {
        "comparisons": [c for c in comparisons if c],
        "source_counts": source_counts,
        "inputs": {
            "base50": args.base50,
            "base40": args.base40,
            "base32": args.base32,
            "temp10": args.temp10,
            "temp8": args.temp8,
            "off10": args.off10,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[PortfolioSummary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
