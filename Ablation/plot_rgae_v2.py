#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from plot_slppo import ROOT, parse_log, _plot_curves

SERIES = [
    ("Exp3 soft gate baseline", "gpu0_exp3_softgate"),
    ("Exp3 + variance gate", "gpu1_exp3_var_gate"),
    ("Exp3 + elite route aux", "gpu2_exp3_elite_route"),
    ("Exp3 + variance + elite", "gpu3_exp3_var_gate_elite_route"),
]


def infer_series(log_dir: Path):
    files = {p.name: p for p in sorted(log_dir.glob("*.txt"))}
    found = {}
    for label, stem in SERIES:
        match = next((path for name, path in files.items() if name.startswith(stem)), None)
        if match is not None:
            found[label] = match
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RGAE-v2 effective rollout selection curves.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eval-freq", type=int, default=10)
    args = parser.parse_args()
    out_dir = args.out_dir or (ROOT / "IMGS" / args.log_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = {}
    for label, log_path in infer_series(args.log_dir).items():
        pts = parse_log(log_path, label, args.eval_freq)
        parsed[label] = pts
        best = [p for p in pts if p.aggregation == "best_of_8"]
        avg = [p for p in pts if p.aggregation == "trajectory_avg"]
        if best:
            latest = best[-1]
            best_pt = min(best, key=lambda p: p.objective)
            print(f"{label} best_of_8: latest={latest.objective:.3f} best={best_pt.objective:.3f} update={latest.update}")
        if avg:
            latest = avg[-1]
            best_pt = min(avg, key=lambda p: p.objective)
            print(f"{label} trajectory_avg: latest={latest.objective:.3f} best={best_pt.objective:.3f} update={latest.update}")

    csv_path = out_dir / "rgae_v2_eval_points.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "path", "update", "mode", "aggregation", "avg_reward", "objective", "solved_rate", "episodes", "avg_done_step", "avg_cs"])
        for label, pts in parsed.items():
            for p in pts:
                writer.writerow([label, p.path, p.update, p.mode, p.aggregation, p.avg_reward, p.objective, p.solved_rate, p.episodes, p.avg_done_step, p.avg_cs])
    print(f"saved: {csv_path}")

    specs = [
        ("best_of_8_objective.png", "best_of_8", "objective", "Objective (-100 * eval reward, lower is better)", "RGAE-v2 Best-of-8 Objective", (900, 1050)),
        ("trajectory_avg_objective.png", "trajectory_avg", "objective", "Objective (-100 * eval reward, lower is better)", "RGAE-v2 Trajectory-Average Objective", (980, 1120)),
        ("best_of_8_avg_cs.png", "best_of_8", "avg_cs", "Average charging station visits", "RGAE-v2 Best-of-8 CS Usage", None),
        ("trajectory_avg_avg_cs.png", "trajectory_avg", "avg_cs", "Average charging station visits", "RGAE-v2 Trajectory-Average CS Usage", None),
    ]
    for filename, aggregation, metric, ylabel, title, zoom in specs:
        out = out_dir / filename
        if _plot_curves(parsed, out, aggregation=aggregation, metric=metric, ylabel=ylabel, title=title, zoom_ylim=zoom):
            print(f"saved: {out}")
        else:
            print(f"skipped: {out}")


if __name__ == "__main__":
    main()
