#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_RE = re.compile(
    r"\[EvalSummary\]\s+mode=(?P<mode>\S+)\s+episodes=(?P<episodes>\d+)"
    r"\s+avg_reward=(?P<reward>\S+)"
    r"(?:.*?avg_done_step=(?P<done>\S+))?"
    r"(?:.*?avg_cs=(?P<cs>\S+))?"
    r"(?:.*?solved_rate=(?P<solved>\S+))?"
    r"(?:.*?aggregation=(?P<agg>\S+))?"
)

LABELS = {
    "gpu0_vanilla": "Vanilla PPO 2-head",
    "gpu1_refcoef0p05_rho0p05": "coef=0.05 rho=0.05",
    "gpu2_refcoef0p10_rho0p10": "coef=0.10 rho=0.10",
    "gpu3_refcoef0p15_rho0p10": "coef=0.15 rho=0.10",
}


def label_for(path: Path) -> str:
    stem = path.stem
    for key, label in LABELS.items():
        if stem.startswith(key):
            return label
    return stem


def parse_log(path: Path, eval_freq: int) -> list[dict]:
    rows = []
    best_idx = 0
    for line in path.read_text(errors="ignore").splitlines():
        m = EVAL_RE.search(line)
        if not m:
            continue
        mode = m.group("mode")
        agg = m.group("agg") or ("trajectory_avg" if mode.endswith("_avg") else "best_of_8")
        if agg == "best_of_8":
            best_idx += 1
            update = best_idx * eval_freq
        else:
            update = best_idx * eval_freq
        reward = float(m.group("reward"))
        rows.append({
            "label": label_for(path),
            "file": path.name,
            "update": update,
            "mode": mode,
            "aggregation": agg,
            "episodes": int(m.group("episodes")),
            "avg_reward": reward,
            "objective": -100.0 * reward,
            "solved_rate": float(m.group("solved")) if m.group("solved") is not None else math.nan,
            "avg_done_step": float(m.group("done")) if m.group("done") is not None else math.nan,
            "avg_cs": float(m.group("cs")) if m.group("cs") is not None else math.nan,
        })
    return rows


def plot_metric(rows: list[dict], out: Path, aggregation: str, metric: str, ylabel: str, title: str, zoom_objective: bool = False) -> None:
    curves = {}
    for r in rows:
        if r["aggregation"] != aggregation:
            continue
        val = r.get(metric)
        if val is None or not math.isfinite(float(val)):
            continue
        curves.setdefault(r["label"], []).append((r["update"], float(val)))
    if not curves:
        return
    if zoom_objective:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.35, 1.0]})
        main_ax, zoom_ax = axes
        for label, pts in curves.items():
            pts = sorted(pts)
            xs, ys = zip(*pts)
            main_ax.plot(xs, ys, marker="o", label=label)
            zoom_ax.plot(xs, ys, marker="o", label=label)
        main_ax.set_title(title)
        main_ax.set_xlabel("Update")
        main_ax.set_ylabel(ylabel)
        main_ax.grid(True, alpha=0.25)
        main_ax.legend()
        zoom_ax.set_title("Objective zoom: 900-1050")
        zoom_ax.set_xlabel("Update")
        zoom_ax.set_ylabel(ylabel)
        zoom_ax.set_ylim(900, 1050)
        zoom_ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        zoom_path = out.with_name(out.stem + "_zoom.png")
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, pts in curves.items():
            pts = sorted(pts)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_title(title + " | zoom 900-1050")
        ax.set_xlabel("Update")
        ax.set_ylabel(ylabel)
        ax.set_ylim(900, 1050)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(zoom_path, dpi=160)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, pts in curves.items():
        pts = sorted(pts)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_summary(rows: list[dict], out_dir: Path) -> None:
    lines = ["# Ablation Summary", ""]
    for aggregation in ["best_of_8", "trajectory_avg"]:
        lines.append(f"## {aggregation}")
        lines.append("")
        lines.append("| Method | Latest update | Final obj | Best obj | Best update | Final solved |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        by_label = {}
        for r in rows:
            if r["aggregation"] == aggregation:
                by_label.setdefault(r["label"], []).append(r)
        for label, pts in by_label.items():
            pts = sorted(pts, key=lambda x: x["update"])
            latest = pts[-1]
            best = min(pts, key=lambda x: x["objective"])
            lines.append(f"| {label} | {latest['update']} | {latest['objective']:.3f} | {best['objective']:.3f} | {best['update']} | {latest['solved_rate']:.6f} |")
        lines.append("")
        vanilla = by_label.get("Vanilla PPO 2-head")
        if vanilla:
            v_latest = sorted(vanilla, key=lambda x: x["update"])[-1]
            lines.append(f"Vanilla final objective: `{v_latest['objective']:.3f}`")
            lines.append("")
            for label, pts in by_label.items():
                if label == "Vanilla PPO 2-head":
                    continue
                latest = sorted(pts, key=lambda x: x["update"])[-1]
                gap = v_latest["objective"] - latest["objective"]
                lines.append(f"- {label}: gap vs Vanilla final = `{gap:.3f}` (`{gap / v_latest['objective'] * 100:.2f}%`)")
            lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eval-freq", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.log_dir.glob("*.txt")):
        rows.extend(parse_log(path, args.eval_freq))
    csv_path = args.out_dir / "eval_points.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["label", "file", "update", "mode", "aggregation", "episodes", "avg_reward", "objective", "solved_rate", "avg_done_step", "avg_cs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for agg in ["best_of_8", "trajectory_avg"]:
        plot_metric(rows, args.out_dir / f"{agg}_objective.png", agg, "objective", "Objective (-100 * reward), lower is better", f"Ablation {agg} objective", zoom_objective=True)
        plot_metric(rows, args.out_dir / f"{agg}_solved_rate.png", agg, "solved_rate", "Solved rate", f"Ablation {agg} solved rate")
        plot_metric(rows, args.out_dir / f"{agg}_avg_cs.png", agg, "avg_cs", "Average charging stations", f"Ablation {agg} avg CS")
    write_summary(rows, args.out_dir)
    print(f"saved csv: {csv_path}")
    print(f"saved summary: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
