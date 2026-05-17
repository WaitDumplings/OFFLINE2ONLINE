#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda r: int(r["instance_id"]))
    ids = np.array([int(r["instance_id"]) for r in rows], dtype=np.int64)
    obj = np.array([float(r["objective_value"]) for r in rows], dtype=np.float64)
    k = int(rows[0].get("n_traj", 0))
    return ids, obj, k


def safe_name(path):
    name = Path(path).stem
    name = re.sub(r"^solomon_", "", name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--glob", action="append", default=[])
    p.add_argument("--target-k", type=int, default=50)
    p.add_argument("--max-branches", type=int, default=4)
    p.add_argument("--keep-top-per-k", type=int, default=0)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    base_ids, base_obj, _ = read_rows(args.base)
    candidates = []
    seen = {str(Path(args.base).resolve())}
    for pattern in args.glob:
        for path in sorted(Path().glob(pattern)):
            if not path.is_file():
                continue
            if path.name.endswith(".summary.json"):
                continue
            if "portfolio_" in path.name:
                continue
            if path.suffix != ".csv":
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            ids, obj, k = read_rows(path)
            if k <= 0 or k > args.target_k:
                continue
            if len(obj) != len(base_obj) or not np.array_equal(ids, base_ids):
                continue
            seen.add(resolved)
            candidates.append({"name": safe_name(path), "path": str(path), "k": k, "obj": obj})

    if args.keep_top_per_k > 0:
        kept = []
        for k in sorted({c["k"] for c in candidates}):
            group = [c for c in candidates if c["k"] == k]
            group.sort(key=lambda c: float(c["obj"].mean()))
            kept.extend(group[: args.keep_top_per_k])
        candidates = kept

    results = []
    for r in range(1, min(args.max_branches, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, r):
            if sum(c["k"] for c in combo) != args.target_k:
                continue
            stack = np.stack([c["obj"] for c in combo], axis=0)
            best_idx = np.argmin(stack, axis=0)
            best_obj = stack[best_idx, np.arange(stack.shape[1])]
            delta = best_obj - base_obj
            source_counts = {
                combo[idx]["name"]: int((best_idx == idx).sum())
                for idx in range(len(combo))
            }
            results.append(
                {
                    "names": [c["name"] for c in combo],
                    "paths": [c["path"] for c in combo],
                    "ks": [c["k"] for c in combo],
                    "avg": float(best_obj.mean()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "trimmed_mean_delta": float(np.sort(delta)[int(0.05 * len(delta)) : int(0.95 * len(delta))].mean()),
                    "wins": int((delta < -1e-9).sum()),
                    "losses": int((delta > 1e-9).sum()),
                    "ties": int((np.abs(delta) <= 1e-9).sum()),
                    "source_counts": source_counts,
                }
            )

    results.sort(key=lambda x: x["avg"])
    payload = {
        "base": args.base,
        "base_avg": float(base_obj.mean()),
        "target_k": args.target_k,
        "num_candidates": len(candidates),
        "top": results[: args.top],
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
