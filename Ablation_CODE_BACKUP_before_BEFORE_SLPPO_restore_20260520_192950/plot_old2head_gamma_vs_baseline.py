#!/usr/bin/env python3
"""Plot gamma-corrected old-2head PBRS sweep against base-ref baselines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/data/Maojie/Github2/EVRP-TW-D-B_Weekend")
OUT_DIR = ROOT / "imgs"
BASE_DIR = ROOT / "LOGS" / (
    "Codex_Res_base_ref800_repair_progress_objprog_enc512_s75_ntraj50_nm64"
)
GAMMA_DIR = ROOT / "LOGS" / (
    "Codex_Res_pbrs_old2head_gamma200_ne128_mb64_"
    "repair_progress_objprog_enc512_s75_ntraj50_u200"
)


@dataclass
class Series:
    label: str
    path: Path
    color: str
    linestyle: str = "-"
    linewidth: float = 1.8
    alpha: float = 1.0


EPOCH_RE = re.compile(r"Epoch:\s*(\d+)/(\d+)")
AVG_CUSTOMER_RE = re.compile(r"Avg Customer Visits:\s*([-+0-9.eE]+)")
FINISH_RE = re.compile(r"Finish Rate:\s*(\d+)/(\d+)\s*=\s*([-+0-9.eE]+)")
EVAL_RE = re.compile(
    r"\[EvalSummary\]\s+mode=pomo\s+episodes=(\d+)\s+avg_reward=([-+0-9.eE]+)"
    r".*?solved_rate=([-+0-9.eE]+)"
)


def one(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"missing log for pattern={pattern} root={root}")
    return matches[-1]


SERIES = [
    Series(
        "Baseline seed2025",
        one("result_gpu2_*seed2025_u800.txt", BASE_DIR),
        "#333333",
        "--",
        2.2,
        0.85,
    ),
    Series(
        "Baseline best seed2030",
        one("result_gpu1_*seed2030_u800.txt", BASE_DIR),
        "#777777",
        ":",
        2.2,
        0.9,
    ),
    Series(
        "Gamma c=2.0 beta=0.75",
        one("result_gpu0_*pbrsG_c2p0_b0p75*.txt", GAMMA_DIR),
        "#1f77b4",
    ),
    Series(
        "Gamma c=2.0 beta=1.00",
        one("result_gpu1_*pbrsG_c2p0_b1p00*.txt", GAMMA_DIR),
        "#ff7f0e",
    ),
    Series(
        "Gamma c=1.0 beta=0.75",
        one("result_gpu2_*pbrsG_c1p0_b0p75*.txt", GAMMA_DIR),
        "#2ca02c",
    ),
    Series(
        "Gamma c=1.0 beta=1.00",
        one("result_gpu3_*pbrsG_c1p0_b1p00*.txt", GAMMA_DIR),
        "#d62728",
    ),
]


def parse_log(path: Path) -> dict[str, list[tuple[int, float]]]:
    train_finish: list[tuple[int, float]] = []
    train_customer: list[tuple[int, float]] = []
    eval_objective: list[tuple[int, float]] = []
    eval_solved: list[tuple[int, float]] = []

    current_epoch: int | None = None
    pending_avg_customer: float | None = None
    for line in path.read_text(errors="ignore").splitlines():
        m_epoch = EPOCH_RE.search(line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))
            pending_avg_customer = None
            continue

        if current_epoch is None:
            continue

        m_customer = AVG_CUSTOMER_RE.search(line)
        if m_customer:
            pending_avg_customer = float(m_customer.group(1))
            continue

        m_finish = FINISH_RE.search(line)
        if m_finish:
            finish_rate = float(m_finish.group(3))
            train_finish.append((current_epoch, finish_rate))
            if pending_avg_customer is not None:
                train_customer.append((current_epoch, pending_avg_customer))
            continue

        m_eval = EVAL_RE.search(line)
        if m_eval:
            avg_reward = float(m_eval.group(2))
            solved_rate = float(m_eval.group(3))
            eval_objective.append((current_epoch, -100.0 * avg_reward))
            eval_solved.append((current_epoch, solved_rate))

    return {
        "train_finish": train_finish,
        "train_customer": train_customer,
        "eval_objective": eval_objective,
        "eval_solved": eval_solved,
    }


def plot_points(ax, points: list[tuple[int, float]], series: Series, marker: str = "") -> None:
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(
        xs,
        ys,
        label=series.label,
        color=series.color,
        linestyle=series.linestyle,
        linewidth=series.linewidth,
        alpha=series.alpha,
        marker=marker,
        markersize=4,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed = {series.label: parse_log(series.path) for series in SERIES}

    print("Latest parsed points:")
    for series in SERIES:
        data = parsed[series.label]
        finish = data["train_finish"][-1] if data["train_finish"] else None
        eval_obj = data["eval_objective"][-1] if data["eval_objective"] else None
        best_obj = min(data["eval_objective"], key=lambda x: x[1]) if data["eval_objective"] else None
        print(
            f"{series.label:25s} | train_finish={finish} | "
            f"latest_eval_obj={eval_obj} | best_eval_obj={best_obj}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    ax_finish, ax_customer, ax_obj = axes

    for series in SERIES:
        data = parsed[series.label]
        plot_points(ax_finish, data["train_finish"], series)
        plot_points(ax_customer, data["train_customer"], series)
        plot_points(ax_obj, data["eval_objective"], series, marker="o")

    ax_finish.set_title("Training Progress vs Base-Ref Baseline")
    ax_finish.set_ylabel("Train finish rate")
    ax_finish.set_ylim(-0.03, 1.03)
    ax_finish.grid(True, alpha=0.3)

    ax_customer.set_ylabel("Train avg customers")
    ax_customer.set_ylim(20, 50.5)
    ax_customer.grid(True, alpha=0.3)

    ax_obj.set_ylabel("Eval objective proxy (-100 * avg_reward)")
    ax_obj.set_xlabel("Training update")
    ax_obj.grid(True, alpha=0.3)

    handles, labels = ax_finish.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "pbrs_old2head_gamma_vs_baseline_progress.png"
    fig.savefig(out, dpi=180)
    print(f"saved: {out}")

    # Early zoom makes the current 200-update sweep readable against 800-update baselines.
    fig_zoom, axes_zoom = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    z_finish, z_customer, z_obj = axes_zoom
    for series in SERIES:
        data = parsed[series.label]
        plot_points(z_finish, [p for p in data["train_finish"] if p[0] <= 80], series)
        plot_points(z_customer, [p for p in data["train_customer"] if p[0] <= 80], series)
        plot_points(z_obj, [p for p in data["eval_objective"] if p[0] <= 80], series, marker="o")

    z_finish.set_title("Training Progress vs Base-Ref Baseline (0-80 updates)")
    z_finish.set_ylabel("Train finish rate")
    z_finish.set_ylim(-0.03, 1.03)
    z_finish.grid(True, alpha=0.3)
    z_customer.set_ylabel("Train avg customers")
    z_customer.set_ylim(20, 50.5)
    z_customer.grid(True, alpha=0.3)
    z_obj.set_ylabel("Eval objective proxy (-100 * avg_reward)")
    z_obj.set_xlabel("Training update")
    z_obj.grid(True, alpha=0.3)
    handles, labels = z_finish.get_legend_handles_labels()
    fig_zoom.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig_zoom.tight_layout(rect=(0, 0, 1, 0.93))
    out_zoom = OUT_DIR / "pbrs_old2head_gamma_vs_baseline_progress_zoom80.png"
    fig_zoom.savefig(out_zoom, dpi=180)
    print(f"saved: {out_zoom}")


if __name__ == "__main__":
    main()
