#!/usr/bin/env python3
"""Filter selected missing-mode records by improvement and novelty criteria."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument("--min-mean-nll", type=float, default=0.0)
    parser.add_argument("--min-lowprob-ratio", type=float, default=0.0)
    parser.add_argument("--top-novelty-frac", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.input_pkl, "rb") as f:
        records = pickle.load(f)

    threshold_nll = float(args.min_mean_nll)
    if float(args.top_novelty_frac) > 0:
        nll = np.asarray(
            [float(r["mode_proposal"].get("mean_nll", 0.0)) for r in records],
            dtype=np.float64,
        )
        q = max(0.0, min(1.0, 1.0 - float(args.top_novelty_frac)))
        threshold_nll = max(threshold_nll, float(np.quantile(nll, q)))

    out = []
    for record in records:
        meta = record.get("mode_proposal", {})
        if float(meta.get("improvement_rel", 0.0)) < float(args.min_improvement):
            continue
        if float(meta.get("mean_nll", 0.0)) < threshold_nll:
            continue
        if float(meta.get("lowprob_ratio", 0.0)) < float(args.min_lowprob_ratio):
            continue
        out.append(record)

    Path(args.output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(out, f)
    summary = {
        "input_records": len(records),
        "output_records": len(out),
        "min_improvement": float(args.min_improvement),
        "min_mean_nll": float(args.min_mean_nll),
        "effective_mean_nll_threshold": threshold_nll,
        "min_lowprob_ratio": float(args.min_lowprob_ratio),
        "top_novelty_frac": float(args.top_novelty_frac),
        "output_pkl": args.output_pkl,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[FilterMissingMode] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
