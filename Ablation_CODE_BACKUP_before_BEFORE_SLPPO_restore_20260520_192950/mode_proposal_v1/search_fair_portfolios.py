#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def read_objectives(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["instance_id"]), float(row["objective_value"])))
    rows.sort(key=lambda x: x[0])
    ids = np.array([x[0] for x in rows], dtype=np.int64)
    obj = np.array([x[1] for x in rows], dtype=np.float64)
    return ids, obj


def parse_candidate(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"candidate must be name:k:path, got {spec}")
    name, k_str, path = parts
    return {"name": name, "k": int(k_str), "path": path}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", action="append", default=[])
    p.add_argument("--target-k", type=int, default=50)
    p.add_argument("--max-branches", type=int, default=4)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    base_ids, base_obj = read_objectives(args.base)
    base_mean = float(base_obj.mean())
    candidates = []
    for spec in args.candidate:
        c = parse_candidate(spec)
        ids, obj = read_objectives(c["path"])
        if not np.array_equal(ids, base_ids):
            raise ValueError(f"instance ids mismatch for {c['name']}: {c['path']}")
        c["obj"] = obj
        candidates.append(c)

    results = []
    for r in range(1, min(args.max_branches, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, r):
            total_k = sum(c["k"] for c in combo)
            if total_k != args.target_k:
                continue
            stack = np.stack([c["obj"] for c in combo], axis=0)
            best_idx = np.argmin(stack, axis=0)
            best_obj = stack[best_idx, np.arange(stack.shape[1])]
            delta = best_obj - base_obj
            source_counts = {}
            for idx, c in enumerate(combo):
                source_counts[c["name"]] = int((best_idx == idx).sum())
            results.append(
                {
                    "names": [c["name"] for c in combo],
                    "paths": [c["path"] for c in combo],
                    "total_k": total_k,
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
        "base_avg": base_mean,
        "target_k": args.target_k,
        "top": results[: args.top],
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
