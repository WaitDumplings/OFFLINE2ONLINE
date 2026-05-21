#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMON = [
    "--cuda", "true",
    "--torch-deterministic", "true",
    "--eval-data-path", str(ROOT / "dataset" / "unanchored" / "Cus_50" / "pickle" / "evrptw_50C_12R.pkl"),
    "--eval-freq", "10",
    "--eval-batch-size", "1000",
    "--num-envs", "128",
    "--num-steps", "75",
    "--n-traj", "50",
    "--test-agent", "8",
    "--num-minibatches", "128",
    "--accum-steps", "16",
    "--eval-decode-mode", "sampling",
    "--train-config-schedule", "batch_cycle",
    "--train-config-stratify-keys", "instance_type,time_window_policy",
    "--train-config-fixed-overrides", "service_time_policy=cargoweight,cluster_number_policy=random",
    "--train-config-use-online-counter", "true",
    "--train-env-seed-by-update", "true",
    "--use-direct-progress-pbrs", "true",
    "--progress-pbrs-coef", "2.0",
    "--progress-pbrs-beta", "0.5",
    "--use-repair-fail-reward", "true",
    "--repair-fail-coef", "1.0",
    "--repair-progress-coef", "1.0",
    "--repair-success-bonus", "1.0",
    "--use-decomposed-reward-adv", "true",
    "--decomposed-reward-mode", "objective_progress",
    "--adv-objective-weight", "0.5",
    "--adv-progress-weight", "0.5",
    "--debug", "true",
    "--debug-test", "true",
]

P5_BEST_BASE = [
    "--adv-weight-schedule", "fixed",
    "--alns-buffer-dir", str(ROOT / "dataset" / "unanchored" / "Cus_50" / "buffer"),
    "--incumbent-ratio", "0.10",
    "--buffer-normal-frac", "1.0",
    "--buffer-temp-frac", "0.0",
    "--buffer-prefix-frac", "0.0",
    "--online-normal-frac", "1.0",
    "--online-temp-frac", "0.0",
    "--temperature-sampling", "1.0",
    "--use-regret-aware-sampler", "true",
    "--use-pomo50-regret-state", "true",
    "--pomo50-stable-beat-rate", "0.20",
    "--pomo50-low-beat-rate", "0.05",
    "--pomo50-regret-margin-rel", "0.01",
    "--buffer-regret-relative", "true",
    "--buffer-regret-margin", "0.0",
    "--buffer-regret-kappa", "25.0",
    "--buffer-unknown-weight", "1.00",
    "--buffer-uncertain-weight", "0.50",
    "--buffer-ppo-win-weight", "0.05",
    "--buffer-lucky-ppo-weight", "0.35",
    "--buffer-alns-win-base-weight", "1.0",
    "--regret-recompute-freq", "10",
    "--regret-probe-size", "256",
    "--use-route-preference", "false",
    "--use-selective-bc", "false",
    "--use-self-generated-buffer", "false",
    "--cmp-adv-coef", "0.0",
    "--adv-diag", "true",
    "--grad-cos-diag", "true",
    "--grad-cos-freq", "10",
    "--group-adv-coef", "0.30",
    "--group-adv-clip", "3.0",
    "--reference-adv-coef", "0.10",
    "--reference-adv-rho", "0.10",
    "--reference-adv-clip", "2.0",
    "--reference-adv-gate-temp", "0.05",
    "--reference-adv-hard-threshold", "0.03",
    "--reference-adv-source", "best_archive",
    "--allow-incumbent-updates", "true",
    "--reference-adv-alns-win-only", "false",
    "--reference-adv-allow-states", "",
    "--reference-adv-gate-mode", "linear",
]


def milestone_args(coef: float, rho: float, clip: float = 2.0) -> list[str]:
    return [
        "--use-milestone-reference", "true",
        "--milestone-ref-coef", f"{coef:.2f}",
        "--milestone-ref-rho", f"{rho:.2f}",
        "--milestone-ref-clip", f"{clip:.1f}",
        "--milestone-ref-denom", "milestone",
        "--milestone-ref-use-ref-gate", "true",
    ]


def build_jobs(seed: int, num_updates: int, save_dir: Path) -> list[dict]:
    common = ["--seed", str(seed), "--num-updates", str(num_updates), "--save-dir", str(save_dir), *COMMON]
    return [
        {
            "label": "P7-GPU0 control: P5 best sampler, no milestone reference",
            "name": f"gpu0_p7_control_p5best_seed{seed}_u{num_updates}",
            "gpu": "0",
            "p7_note": "control: P5 best_archive + group/ref + low archive ratio; milestone disabled",
            "extra": [*P5_BEST_BASE, "--use-milestone-reference", "false", "--milestone-ref-coef", "0.0"],
            "common": common,
        },
        {
            "label": "P7-GPU1 weak milestone",
            "name": f"gpu1_p7_milestone_c005_r010_seed{seed}_u{num_updates}",
            "gpu": "1",
            "p7_note": "milestone_ref_coef=0.05, rho=0.10",
            "extra": [*P5_BEST_BASE, *milestone_args(coef=0.05, rho=0.10)],
            "common": common,
        },
        {
            "label": "P7-GPU2 medium milestone",
            "name": f"gpu2_p7_milestone_c010_r010_seed{seed}_u{num_updates}",
            "gpu": "2",
            "p7_note": "milestone_ref_coef=0.10, rho=0.10",
            "extra": [*P5_BEST_BASE, *milestone_args(coef=0.10, rho=0.10)],
            "common": common,
        },
        {
            "label": "P7-GPU3 sharper milestone",
            "name": f"gpu3_p7_milestone_c010_r005_seed{seed}_u{num_updates}",
            "gpu": "3",
            "p7_note": "milestone_ref_coef=0.10, rho=0.05",
            "extra": [*P5_BEST_BASE, *milestone_args(coef=0.10, rho=0.05)],
            "common": common,
        },
    ]


def check_required_files() -> None:
    required = [
        ROOT / "train_incumbent.py",
        ROOT / "plot_ablation.py",
        ROOT / "dataset" / "unanchored" / "Cus_50" / "pickle" / "evrptw_50C_12R.pkl",
        ROOT / "dataset" / "unanchored" / "Cus_50" / "buffer" / "progress" / "buffer_progress.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch P7 milestone-reference experiments on GPU0-3.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3004")))
    parser.add_argument("--num-updates", type=int, default=int(os.environ.get("NUM_UPDATES", "300")))
    parser.add_argument("--python-bin", type=str, default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    check_required_files()
    run_name = f"p7_milestone_seed{args.seed}_u{args.num_updates}"
    log_dir = ROOT / "LOGS" / run_name
    img_dir = ROOT / "IMGS" / run_name
    save_dir = ROOT / "checkpoint" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.seed, args.num_updates, save_dir)
    metadata = {
        "seed": args.seed,
        "num_updates": args.num_updates,
        "root": str(ROOT),
        "log_dir": str(log_dir),
        "img_dir": str(img_dir),
        "save_dir": str(save_dir),
        "p7_base": {
            "purpose": "Add progress-aligned reference milestones on top of the current P5 best sampler.",
            "control": "P5 low archive ratio best_archive group/ref without milestone.",
            "reference_adv_source": "best_archive",
            "reference_adv_gate_mode": "linear",
            "reference_adv_coef": 0.10,
            "reference_adv_rho": 0.10,
            "group_adv_coef": 0.30,
            "incumbent_ratio": 0.10,
            "n_traj": 50,
            "test_agent": 8,
        },
        "jobs": [],
    }

    processes = []
    for job in jobs:
        log_path = log_dir / f"{job['name']}.txt"
        cmd = [args.python_bin, "train_incumbent.py", "--exp-name", job["name"], "--cuda-id", job["gpu"], *job["common"], *job["extra"]]
        job_meta = {
            "label": job["label"],
            "name": job["name"],
            "gpu": job["gpu"],
            "p7_note": job["p7_note"],
            "log": str(log_path),
            "cmd": cmd,
        }
        metadata["jobs"].append(job_meta)
        print(f"[launch] gpu{job['gpu']} {job['name']} -> {log_path}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        fh = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        job_meta["pid"] = proc.pid
        processes.append((job_meta, proc, fh))

    (log_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.dry_run:
        print(f"[dry-run] metadata written to {log_dir / 'run_metadata.json'}", flush=True)
        return 0

    print(f"[wait] launched {len(processes)} jobs. Metadata: {log_dir / 'run_metadata.json'}", flush=True)
    alive = {m["name"]: True for m, _, _ in processes}
    while True:
        remaining = []
        for meta, proc, _fh in processes:
            code = proc.poll()
            if code is None:
                remaining.append(meta["name"])
            elif alive.get(meta["name"], False):
                alive[meta["name"]] = False
                print(f"[done] {meta['name']} returncode={code}", flush=True)
        if not remaining:
            break
        print(f"[wait] still running: {', '.join(remaining)}", flush=True)
        time.sleep(60)

    rc = 0
    for meta, proc, fh in processes:
        fh.close()
        code = proc.returncode
        meta["returncode"] = code
        if code != 0:
            rc = code or 1
    (log_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("[plot] generating plots and summary", flush=True)
    plot_cmd = [args.python_bin, str(ROOT / "plot_ablation.py"), "--log-dir", str(log_dir), "--out-dir", str(img_dir)]
    plot_rc = subprocess.run(plot_cmd, cwd=ROOT).returncode
    if plot_rc != 0 and rc == 0:
        rc = plot_rc
    print(f"[complete] rc={rc} logs={log_dir} imgs={img_dir}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
