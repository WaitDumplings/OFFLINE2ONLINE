#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


# e.g.
# Evaluation over 10 episodes: -44.034, Step: 131, Avg Done Step: 126.90, #CS visited: 15.00
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PATTERN = re.compile(
    rf"Evaluation over\s+\d+\s+episodes:\s*({FLOAT})\s*,\s*"
    rf"Step:\s*({FLOAT})\s*,\s*"
    rf"Avg Done Step:\s*({FLOAT})\s*,\s*"
    rf"#CS visited:\s*({FLOAT})"
)


def parse_file(txt_path: Path):
    evals, steps, avg_done_steps, cs_visited = [], [], [], []

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            evals.append(float(m.group(1)))
            steps.append(float(m.group(2)))
            avg_done_steps.append(float(m.group(3)))
            cs_visited.append(float(m.group(4)))

    return evals, steps, avg_done_steps, cs_visited


def plot_and_save(x, y, title, ylabel, out_path: Path, show: bool):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Record Index (occurrence order in log)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to the .txt log file")
    ap.add_argument("--outdir", default="eval_plots", help="Directory to save plots")
    ap.add_argument("--show", action="store_true", help="Show figures interactively")
    args = ap.parse_args()

    txt_path = Path(args.log).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    evals, steps, avg_done_steps, cs_visited = parse_file(txt_path)

    if not evals:
        print(f"[ERROR] No 'Evaluation over ...' lines found in: {txt_path}")
        return

    x = list(range(1, len(evals) + 1))

    plot_and_save(x, evals, "Evaluation Score", "Eval (reward)", outdir / "eval.png", args.show)
    plot_and_save(x, steps, "Step", "Step", outdir / "step.png", args.show)
    plot_and_save(x, avg_done_steps, "Avg Done Step", "Avg Done Step", outdir / "avg_done_step.png", args.show)
    plot_and_save(x, cs_visited, "#CS Visited", "#CS visited", outdir / "cs_visited.png", args.show)

    print("[OK] Parsed records:", len(evals))
    print("[OK] Saved plots to:", outdir)


if __name__ == "__main__":
    main()
