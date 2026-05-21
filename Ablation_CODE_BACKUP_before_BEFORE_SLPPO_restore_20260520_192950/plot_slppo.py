#!/usr/bin/env python3
"""Plot SLPPO/Exp3 soft-gate eval curves."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

EVAL_RE = re.compile(
    r"\[EvalSummary\]\s+mode=(?P<mode>\S+)\s+episodes=(?P<episodes>\d+)"
    r"\s+avg_reward=(?P<reward>\S+)"
    r"(?:.*?avg_done_step=(?P<done>\S+))?"
    r"(?:.*?avg_cs=(?P<cs>\S+))?"
    r"(?:.*?solved_rate=(?P<solved>\S+))?"
    r"(?:.*?aggregation=(?P<aggregation>\S+))?"
)


@dataclass(frozen=True)
class EvalPoint:
    label: str
    update: int
    mode: str
    aggregation: str
    avg_reward: float
    objective: float
    solved_rate: float | None
    episodes: int
    avg_done_step: float | None
    avg_cs: float | None
    path: Path


def _infer_series(log_dir: Path) -> dict[str, Path]:
    files = {p.name: p for p in sorted(log_dir.glob("*.txt"))}
    ordered: list[tuple[str, str]] = [
        ("Vanilla PPO 2-head", "gpu0_vanilla_ppo_2head"),
        ("Exp3 soft gate", "gpu1_exp3_softgate_seed"),
        ("Exp3 soft gate + SL-PPO medium", "gpu2_exp3_softgate_slppo_m"),
        ("Exp3 soft gate + SL-PPO strong", "gpu3_exp3_softgate_slppo_s"),
    ]
    series: dict[str, Path] = {}
    for label, stem in ordered:
        match = next((path for name, path in files.items() if name.startswith(stem)), None)
        if match is not None:
            series[label] = match
    return series


def parse_log(path: Path, label: str, eval_freq: int) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    primary_index = 0
    last_primary_update: int | None = None
    for line in path.read_text(errors="ignore").splitlines():
        m = EVAL_RE.search(line)
        if not m:
            continue
        mode = m.group("mode")
        aggregation = m.group("aggregation") or ("trajectory_avg" if mode.endswith("_avg") else "best_of_8")
        if aggregation == "trajectory_avg" or mode.endswith("_avg"):
            update = last_primary_update if last_primary_update is not None else (primary_index + 1) * eval_freq
        else:
            primary_index += 1
            update = primary_index * eval_freq
            last_primary_update = update
        reward = float(m.group("reward"))
        solved_raw = m.group("solved")
        done_raw = m.group("done")
        cs_raw = m.group("cs")
        points.append(
            EvalPoint(
                label=label,
                update=update,
                mode=mode,
                aggregation=aggregation,
                avg_reward=reward,
                objective=-100.0 * reward,
                solved_rate=float(solved_raw) if solved_raw is not None else None,
                episodes=int(m.group("episodes")),
                avg_done_step=float(done_raw) if done_raw is not None else None,
                avg_cs=float(cs_raw) if cs_raw is not None else None,
                path=path,
            )
        )
    return points


def _metric_points(points: list[EvalPoint], aggregation: str, metric: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for pt in points:
        if pt.aggregation != aggregation:
            continue
        value = getattr(pt, metric)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        xs.append(pt.update)
        ys.append(float(value))
    return xs, ys


def _plot_curves(
    parsed: dict[str, list[EvalPoint]],
    out_path: Path,
    *,
    aggregation: str,
    metric: str,
    ylabel: str,
    title: str,
    zoom_ylim: tuple[float, float] | None = None,
) -> bool:
    curves: dict[str, tuple[list[int], list[float]]] = {}
    for label, points in parsed.items():
        xs, ys = _metric_points(points, aggregation, metric)
        if xs:
            curves[label] = (xs, ys)
    if not curves:
        return False

    if zoom_ylim is not None:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14.0, 5.4),
            gridspec_kw={"width_ratios": [1.35, 1.0]},
        )
        main_ax, zoom_ax = axes
        for label, (xs, ys) in curves.items():
            main_ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.0, label=label)
            zoom_ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.0, label=label)
        main_ax.set_title(title)
        main_ax.set_xlabel("Training update")
        main_ax.set_ylabel(ylabel)
        main_ax.grid(True, alpha=0.3)
        main_ax.legend()
        zoom_ax.set_title(f"Zoom: {zoom_ylim[0]:.0f}-{zoom_ylim[1]:.0f}")
        zoom_ax.set_xlabel("Training update")
        zoom_ax.set_ylabel(ylabel)
        zoom_ax.set_ylim(*zoom_ylim)
        zoom_ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)

        zoom_path = out_path.with_name(out_path.stem + "_zoom.png")
        zoom_fig, zoom_only_ax = plt.subplots(figsize=(9.5, 5.5))
        for label, (xs, ys) in curves.items():
            zoom_only_ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.0, label=label)
        zoom_only_ax.set_title(f"{title} Zoom")
        zoom_only_ax.set_xlabel("Training update")
        zoom_only_ax.set_ylabel(ylabel)
        zoom_only_ax.set_ylim(*zoom_ylim)
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
        ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.0, label=label)
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
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eval-freq", type=int, default=10)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ROOT / "IMGS" / args.log_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    series = _infer_series(args.log_dir)
    parsed: dict[str, list[EvalPoint]] = {}
    print(f"log_dir: {args.log_dir}")
    for label, path in series.items():
        points = parse_log(path, label, args.eval_freq)
        parsed[label] = points
        best_points = [p for p in points if p.aggregation == "best_of_8"]
        avg_points = [p for p in points if p.aggregation == "trajectory_avg"]
        if best_points:
            latest = best_points[-1]
            best = min(best_points, key=lambda p: p.objective)
            print(f"{label} best_of_8: latest={latest.objective:.3f} best={best.objective:.3f} update={latest.update}")
        if avg_points:
            latest_avg = avg_points[-1]
            best_avg = min(avg_points, key=lambda p: p.objective)
            print(f"{label} trajectory_avg: latest={latest_avg.objective:.3f} best={best_avg.objective:.3f} update={latest_avg.update}")

    csv_path = out_dir / "slppo_eval_points.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "path",
                "update",
                "mode",
                "aggregation",
                "avg_reward",
                "objective",
                "solved_rate",
                "episodes",
                "avg_done_step",
                "avg_cs",
            ]
        )
        for label, points in parsed.items():
            for p in points:
                writer.writerow(
                    [
                        label,
                        p.path,
                        p.update,
                        p.mode,
                        p.aggregation,
                        p.avg_reward,
                        p.objective,
                        p.solved_rate,
                        p.episodes,
                        p.avg_done_step,
                        p.avg_cs,
                    ]
                )
    print(f"saved: {csv_path}")

    plot_specs = [
        (
            "best_of_8_objective.png",
            "best_of_8",
            "objective",
            "Objective (-100 * eval reward, lower is better)",
            "SLPPO Seed Eval: Best-of-8 Objective",
            (900, 1050),
        ),
        (
            "trajectory_avg_objective.png",
            "trajectory_avg",
            "objective",
            "Objective (-100 * eval reward, lower is better)",
            "SLPPO Seed Eval: Trajectory-Average Objective",
            (980, 1120),
        ),
        (
            "best_of_8_reward.png",
            "best_of_8",
            "avg_reward",
            "Eval reward (higher is better)",
            "SLPPO Seed Eval: Best-of-8 Reward",
            None,
        ),
        (
            "trajectory_avg_reward.png",
            "trajectory_avg",
            "avg_reward",
            "Eval reward (higher is better)",
            "SLPPO Seed Eval: Trajectory-Average Reward",
            None,
        ),
        (
            "best_of_8_avg_cs.png",
            "best_of_8",
            "avg_cs",
            "Average charging station visits",
            "SLPPO Seed Eval: Best-of-8 CS Usage",
            None,
        ),
        (
            "trajectory_avg_avg_cs.png",
            "trajectory_avg",
            "avg_cs",
            "Average charging station visits",
            "SLPPO Seed Eval: Trajectory-Average CS Usage",
            None,
        ),
        (
            "best_of_8_solved_rate.png",
            "best_of_8",
            "solved_rate",
            "Solved rate",
            "SLPPO Seed Eval: Best-of-8 Solved Rate",
            None,
        ),
        (
            "trajectory_avg_solved_rate.png",
            "trajectory_avg",
            "solved_rate",
            "Solved rate",
            "SLPPO Seed Eval: Trajectory-Average Solved Rate",
            None,
        ),
    ]
    for filename, aggregation, metric, ylabel, title, zoom in plot_specs:
        out_path = out_dir / filename
        if _plot_curves(
            parsed,
            out_path,
            aggregation=aggregation,
            metric=metric,
            ylabel=ylabel,
            title=title,
            zoom_ylim=zoom,
        ):
            print(f"saved: {out_path}")
        else:
            print(f"skipped: {out_path}")


if __name__ == "__main__":
    main()
