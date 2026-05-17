#!/usr/bin/env python3
"""Plot RGAE-PPO eval curves from LOGS_NEW into IMGS_NEW."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/data/Maojie/Github2/EVRP-TW-D-B_Weekend")
DEFAULT_LOG_DIR = ROOT / "LOGS_NEW" / "RGAE_1000_seed3000"
DEFAULT_OUT_DIR = ROOT / "IMGS_NEW" / "RGAE_1000_seed3000"

SERIES = {
    "Vanilla PPO 2-head": "gpu0_vanilla_ppo_2head_seed3000_u1000.txt",
    "RGAE Exp1 sampler": "gpu1_rgae_exp1_sampler_seed3000_u1000.txt",
    "RGAE Exp2 sampler+group": "gpu2_rgae_exp2_sampler_group_seed3000_u1000.txt",
    "RGAE Exp3 sampler+group+ref": "gpu3_rgae_exp3_sampler_group_ref_seed3000_u1000.txt",
}

EPOCH_RE = re.compile(r"Epoch:\s*(\d+)/(\d+)")
TRAIN_RE = re.compile(r"\[Train\]\s+update=(\d+)\b")
EVAL_RE = re.compile(
    r"\[EvalSummary\]\s+mode=(?P<mode>\S+)\s+episodes=(?P<episodes>\d+)"
    r"\s+avg_reward=(?P<reward>\S+)"
    r"(?:.*?avg_done_step=(?P<done>\S+))?"
    r"(?:.*?avg_cs=(?P<cs>\S+))?"
    r"(?:.*?solved_rate=(?P<solved>\S+))?"
)


@dataclass(frozen=True)
class EvalPoint:
    label: str
    update: int
    mode: str
    avg_reward: float
    objective_proxy: float
    solved_rate: float | None
    episodes: int
    avg_done_step: float | None
    avg_cs: float | None
    path: Path


def parse_log(path: Path, include_trajectory_avg: bool, eval_freq: int) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    eval_index = 0
    last_primary_update: int | None = None
    for line in path.read_text(errors="ignore").splitlines():
        eval_match = EVAL_RE.search(line)
        if not eval_match:
            continue
        mode = eval_match.group("mode")
        is_avg = mode.endswith("_avg")
        if is_avg:
            update = last_primary_update if last_primary_update is not None else (eval_index + 1) * eval_freq
        else:
            eval_index += 1
            update = eval_index * eval_freq
            last_primary_update = update
        if not include_trajectory_avg and is_avg:
            continue
        reward = float(eval_match.group("reward"))
        solved_raw = eval_match.group("solved")
        done_raw = eval_match.group("done")
        cs_raw = eval_match.group("cs")
        points.append(
            EvalPoint(
                label="",
                update=update,
                mode=mode,
                avg_reward=reward,
                objective_proxy=-100.0 * reward,
                solved_rate=float(solved_raw) if solved_raw is not None else None,
                episodes=int(eval_match.group("episodes")),
                avg_done_step=float(done_raw) if done_raw is not None else None,
                avg_cs=float(cs_raw) if cs_raw is not None else None,
                path=path,
            )
        )
    return points


def _collect_metric(parsed: dict[str, list[EvalPoint]], metric: str) -> dict[str, tuple[list[int], list[float]]]:
    curves: dict[str, tuple[list[int], list[float]]] = {}
    for label, pts in parsed.items():
        xs, ys = [], []
        for pt in pts:
            val = getattr(pt, metric)
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                continue
            xs.append(pt.update)
            ys.append(float(val))
        if xs:
            curves[label] = (xs, ys)
    return curves


def _plot(parsed: dict[str, list[EvalPoint]], out_path: Path, metric: str, ylabel: str, title: str) -> bool:
    curves = _collect_metric(parsed, metric)
    if not curves:
        return False

    if metric == "objective_proxy":
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
        main_ax, zoom_ax = axes
        for label, (xs, ys) in curves.items():
            main_ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.2, label=label)
            zoom_ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.2, label=label)
        main_ax.set_title(title)
        main_ax.set_xlabel("Training update")
        main_ax.set_ylabel(ylabel)
        main_ax.grid(True, alpha=0.3)
        main_ax.legend()
        zoom_ax.set_title("Objective zoom: 900-1050")
        zoom_ax.set_xlabel("Training update")
        zoom_ax.set_ylabel("Objective proxy")
        zoom_ax.set_ylim(900, 1050)
        zoom_ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)

        zoom_path = out_path.with_name(out_path.stem + "_zoom.png")
        zoom_fig, zoom_only_ax = plt.subplots(figsize=(9.5, 5.5))
        for label, (xs, ys) in curves.items():
            zoom_only_ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.2, label=label)
        zoom_only_ax.set_title("RGAE Eval Objective Zoom: 900-1050")
        zoom_only_ax.set_xlabel("Training update")
        zoom_only_ax.set_ylabel("Objective proxy")
        zoom_only_ax.set_ylim(900, 1050)
        zoom_only_ax.grid(True, alpha=0.3)
        zoom_only_ax.legend()
        zoom_fig.tight_layout()
        zoom_fig.savefig(zoom_path, dpi=180)
        plt.close(zoom_fig)
        plt.close(fig)
        print(f"saved: {zoom_path}")
        return True

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Training update")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--include-trajectory-avg", action="store_true")
    ap.add_argument("--eval-freq", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parsed: dict[str, list[EvalPoint]] = {}
    print(f"log_dir: {args.log_dir}")
    for label, filename in SERIES.items():
        path = args.log_dir / filename
        if not path.exists():
            parsed[label] = []
            print(f"{label}: missing {path}")
            continue
        pts = parse_log(path, include_trajectory_avg=args.include_trajectory_avg, eval_freq=args.eval_freq)
        parsed[label] = pts
        if not pts:
            print(f"{label}: no eval points yet | {path}")
            continue
        latest = pts[-1]
        best = min(pts, key=lambda p: p.objective_proxy)
        print(
            f"{label}: latest update={latest.update}, obj={latest.objective_proxy:.3f}, "
            f"reward={latest.avg_reward:.5f}, solved={latest.solved_rate}, "
            f"best update={best.update}, best obj={best.objective_proxy:.3f}, mode={latest.mode}"
        )

    csv_path = args.out_dir / "rgae_eval_points.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "path", "update", "mode", "avg_reward", "objective_proxy", "solved_rate", "episodes", "avg_done_step", "avg_cs"])
        for label, pts in parsed.items():
            for pt in pts:
                w.writerow([label, pt.path, pt.update, pt.mode, pt.avg_reward, pt.objective_proxy, pt.solved_rate, pt.episodes, pt.avg_done_step, pt.avg_cs])
    print(f"saved: {csv_path}")

    plots = [
        ("rgae_objective.png", "objective_proxy", "Objective proxy (-100 * eval reward, lower is better)", "RGAE Eval Objective"),
        ("rgae_reward.png", "avg_reward", "Eval avg_reward (higher is better)", "RGAE Eval Reward"),
        ("rgae_solved_rate.png", "solved_rate", "Solved rate", "RGAE Eval Solved Rate"),
        ("rgae_avg_cs.png", "avg_cs", "Average charging station visits", "RGAE Eval CS Usage"),
    ]
    for name, metric, ylabel, title in plots:
        out = args.out_dir / name
        if _plot(parsed, out, metric, ylabel, title):
            print(f"saved: {out}")
        else:
            print(f"skipped: {out}")


if __name__ == "__main__":
    main()
