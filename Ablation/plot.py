#!/usr/bin/env python3
"""Plot eval curves for the vanilla PPO vs competence-guided PPO runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/data/Maojie/Github2/EVRP-TW-D-B_Weekend")
DEFAULT_LOG_DIR = ROOT / "LOGS" / "Competence_Compare_2000_seed3000"
DEFAULT_OUT_DIR = ROOT / "imgs"

SERIES = {
    "Vanilla PPO 2-head seed3000": "result_gpu0_vanilla_ppo_2head_seed3000_u2000.txt",
    "Vanilla PPO 2-head seed3001": "result_gpu1_vanilla_ppo_2head_seed3001_u2000.txt",
    "Competence Full seed3000": "result_gpu2_competence_full_seed3000_u2000.txt",
    "Competence Full seed3001": "result_gpu3_competence_full_seed3001_u2000.txt",
}

EPOCH_RE = re.compile(r"Epoch:\s*(\d+)/(\d+)")
TRAIN_RE = re.compile(r"\[Train\]\s+update=(\d+)\b")
EVAL_RE = re.compile(
    r"\[EvalSummary\]\s+mode=(?P<mode>\S+)\s+episodes=(?P<episodes>\d+)"
    r"\s+avg_reward=(?P<reward>\S+)"
    r"(?:.*?solved_rate=(?P<solved>\S+))?"
)


@dataclass(frozen=True)
class EvalPoint:
    update: int
    mode: str
    avg_reward: float
    objective_proxy: float
    solved_rate: float | None
    episodes: int
    path: Path


def _pick_log(log_dir: Path, pattern: str) -> Path | None:
    matches = sorted(log_dir.glob(pattern))
    return matches[-1] if matches else None


def parse_log(
    path: Path,
    mode_filter: str | None = None,
    include_trajectory_avg: bool = False,
) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    current_update: int | None = None

    for line in path.read_text(errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_update = int(epoch_match.group(1))
            continue

        train_match = TRAIN_RE.search(line)
        if train_match:
            current_update = int(train_match.group(1)) + 1
            continue

        eval_match = EVAL_RE.search(line)
        if not eval_match or current_update is None:
            continue

        mode = eval_match.group("mode")
        if mode_filter and mode != mode_filter:
            continue
        if mode_filter is None and not include_trajectory_avg and mode.endswith("_avg"):
            continue

        avg_reward = float(eval_match.group("reward"))
        solved_raw = eval_match.group("solved")
        solved_rate = float(solved_raw) if solved_raw is not None else None
        points.append(
            EvalPoint(
                update=current_update,
                mode=mode,
                avg_reward=avg_reward,
                objective_proxy=-100.0 * avg_reward,
                solved_rate=solved_rate,
                episodes=int(eval_match.group("episodes")),
                path=path,
            )
        )

    return points


def _write_csv(out_dir: Path, parsed: dict[str, list[EvalPoint]]) -> Path:
    out = out_dir / "competence_compare_eval.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "log_path",
                "update",
                "mode",
                "avg_reward",
                "objective_proxy",
                "solved_rate",
                "episodes",
            ]
        )
        for label, points in parsed.items():
            for point in points:
                writer.writerow(
                    [
                        label,
                        str(point.path),
                        point.update,
                        point.mode,
                        point.avg_reward,
                        point.objective_proxy,
                        "" if point.solved_rate is None else point.solved_rate,
                        point.episodes,
                    ]
                )
    return out


def _plot_metric(
    parsed: dict[str, list[EvalPoint]],
    out_path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> bool:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    any_points = False

    for label, points in parsed.items():
        xs: list[int] = []
        ys: list[float] = []
        for point in points:
            value = getattr(point, metric)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                continue
            xs.append(point.update)
            ys.append(float(value))
        if not xs:
            continue
        any_points = True
        ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.5, label=label)

    if not any_points:
        plt.close(fig)
        return False

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Optional EvalSummary mode filter, for example sampling or pomo.",
    )
    parser.add_argument(
        "--include-trajectory-avg",
        action="store_true",
        help="Include *_avg EvalSummary lines. By default only primary eval modes are plotted.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    parsed: dict[str, list[EvalPoint]] = {}
    print(f"log_dir: {args.log_dir}")
    for label, pattern in SERIES.items():
        path = _pick_log(args.log_dir, pattern)
        if path is None:
            parsed[label] = []
            print(f"{label}: missing ({pattern})")
            continue

        points = parse_log(
            path,
            mode_filter=args.mode,
            include_trajectory_avg=args.include_trajectory_avg,
        )
        parsed[label] = points
        if not points:
            print(f"{label}: no eval points yet | {path}")
            continue

        latest = points[-1]
        best = min(points, key=lambda p: p.objective_proxy)
        print(
            f"{label}: latest update={latest.update}, "
            f"obj={latest.objective_proxy:.3f}, reward={latest.avg_reward:.5f}, "
            f"best update={best.update}, best obj={best.objective_proxy:.3f}, "
            f"mode={latest.mode}, episodes={latest.episodes}"
        )

    csv_path = _write_csv(args.out_dir, parsed)
    print(f"saved: {csv_path}")

    plots = [
        (
            "competence_compare_objective.png",
            "objective_proxy",
            "Objective proxy (-100 * eval avg_reward, lower is better)",
            "Eval Objective Comparison",
        ),
        (
            "competence_compare_reward.png",
            "avg_reward",
            "Eval avg_reward (higher is better)",
            "Eval Reward Comparison",
        ),
        (
            "competence_compare_solved_rate.png",
            "solved_rate",
            "Solved rate",
            "Eval Solved Rate Comparison",
        ),
    ]
    for filename, metric, ylabel, title in plots:
        out_path = args.out_dir / filename
        if _plot_metric(parsed, out_path, metric=metric, ylabel=ylabel, title=title):
            print(f"saved: {out_path}")
        else:
            print(f"skipped: {out_path} (no {metric} points)")


if __name__ == "__main__":
    main()
