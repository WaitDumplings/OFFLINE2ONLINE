#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TRAIN_RE = re.compile(r"\[Train\]\s+update=(\d+)")
EVAL_RE = re.compile(r"\[EvalSummary\].*avg_reward=([-+]?\d+(?:\.\d+)?).*aggregation=(\S+)")


def parse_log(path: Path):
    cur = None
    pending = None
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            cur = int(m.group(1))
        m = EVAL_RE.search(line)
        if not m:
            continue
        obj = -100.0 * float(m.group(1))
        agg = m.group(2)
        if agg == "best_of_8":
            pending = [cur, obj, None]
        elif agg == "trajectory_avg" and pending is not None:
            pending[2] = obj
            rows.append(tuple(pending))
            pending = None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3007)
    parser.add_argument("--num-updates", type=int, default=300)
    parser.add_argument("--zoom-low", type=float, default=960.0)
    parser.add_argument("--zoom-high", type=float, default=1040.0)
    args = parser.parse_args()

    tag = f"slppo_advsplit_seed{args.seed}_u{args.num_updates}"
    log_dir = ROOT / "LOGS" / tag
    img_dir = ROOT / "IMGS" / tag
    img_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        ("Vanilla PPO", f"gpu0_vanilla_ppo_2head_seed{args.seed}_u{args.num_updates}.txt"),
        ("PPO group+ref only", f"gpu1_ppo_archive_only_group_ref_seed{args.seed}_u{args.num_updates}.txt"),
        ("PPO GAE+ref", f"gpu2_ppo_gae_ref_seed{args.seed}_u{args.num_updates}.txt"),
        ("GAE step + SL group+ref", f"gpu3_gae_step_group_ref_slppo_seed{args.seed}_u{args.num_updates}.txt"),
    ]
    data = []
    for label, filename in labels:
        path = log_dir / filename
        rows = parse_log(path) if path.exists() else []
        data.append((label, rows))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=160, constrained_layout=True)
    for label, rows in data:
        if not rows:
            continue
        xs = [r[0] for r in rows]
        best = [r[1] for r in rows]
        axes[0].plot(xs, best, marker="o", linewidth=2, markersize=4, label=label)
        axes[1].plot(xs, best, marker="o", linewidth=2, markersize=4, label=label)
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("Update / epoch")
        ax.set_ylabel("Eval objective, best-of-8 (lower is better)")
        ax.legend(fontsize=8)
    axes[0].set_title(tag)
    axes[1].set_title(f"Zoom {args.zoom_low:.0f}-{args.zoom_high:.0f}")
    axes[1].set_ylim(args.zoom_low, args.zoom_high)
    out = img_dir / f"{tag}_bestof8_zoom.png"
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
