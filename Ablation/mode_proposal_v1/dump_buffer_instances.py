#!/usr/bin/env python3
"""Dump ALNS buffer instances to an eval-compatible pickle."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evrptw_gen.benchmarks.DRL_Solver.train import _load_alns_buffer_from_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-dir", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    records = _load_alns_buffer_from_dir(args.buffer_dir)
    instances = []
    for i, record in enumerate(records):
        inst = dict(record["instance"])
        inst.setdefault("file", record.get("file", f"buffer_{i}.txt"))
        inst.setdefault("instance_id", record.get("instance_id", f"buffer_{i}"))
        instances.append(inst)

    Path(args.output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(instances, f)
    summary = {
        "buffer_dir": args.buffer_dir,
        "instances": len(instances),
        "output_pkl": args.output_pkl,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[DumpBufferInstances] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
