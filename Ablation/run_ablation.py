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

INCUMBENT = [
    "--adv-weight-schedule", "fixed",
    "--alns-buffer-dir", str(ROOT / "dataset" / "unanchored" / "Cus_50" / "buffer"),
    "--incumbent-ratio", "0.20",
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
    "--buffer-regret-kappa", "25.0",
    "--buffer-regret-margin", "0.0",
    "--buffer-unknown-weight", "1.0",
    "--buffer-uncertain-weight", "0.5",
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
    "--reference-adv-alns-win-only", "true",
    "--group-adv-coef", "0.30",
    "--group-adv-clip", "3.0",
    "--reference-adv-clip", "2.0",
]


def build_jobs(seed: int, num_updates: int, save_dir: Path) -> list[dict]:
    common = ["--seed", str(seed), "--num-updates", str(num_updates), "--save-dir", str(save_dir), *COMMON]
    return [
        {
            "label": "Vanilla PPO 2-head",
            "name": f"gpu0_vanilla_ppo_2head_seed{seed}_u{num_updates}",
            "script": "train.py",
            "gpu": "0",
            "extra": ["--use-alns-teacher", "false", "--use-alns-bc", "false", "--use-alns-preference", "false"],
            "common": common,
        },
        {
            "label": "Ref coef=0.05 rho=0.05",
            "name": f"gpu1_refcoef0p05_rho0p05_seed{seed}_u{num_updates}",
            "script": "train_incumbent.py",
            "gpu": "1",
            "extra": [*INCUMBENT, "--reference-adv-coef", "0.05", "--reference-adv-rho", "0.05"],
            "common": common,
        },
        {
            "label": "Ref coef=0.10 rho=0.10",
            "name": f"gpu2_refcoef0p10_rho0p10_seed{seed}_u{num_updates}",
            "script": "train_incumbent.py",
            "gpu": "2",
            "extra": [*INCUMBENT, "--reference-adv-coef", "0.10", "--reference-adv-rho", "0.10"],
            "common": common,
        },
        {
            "label": "Ref coef=0.15 rho=0.10",
            "name": f"gpu3_refcoef0p15_rho0p10_seed{seed}_u{num_updates}",
            "script": "train_incumbent.py",
            "gpu": "3",
            "extra": [*INCUMBENT, "--reference-adv-coef", "0.15", "--reference-adv-rho", "0.10"],
            "common": common,
        },
    ]


def check_required_files() -> None:
    required = [
        ROOT / "train.py",
        ROOT / "train_incumbent.py",
        ROOT / "dataset" / "unanchored" / "Cus_50" / "pickle" / "evrptw_50C_12R.pkl",
        ROOT / "dataset" / "unanchored" / "Cus_50" / "buffer" / "progress" / "buffer_progress.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Ablation ref-coef/rho sweep on GPU0-3.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3003")))
    parser.add_argument("--num-updates", type=int, default=int(os.environ.get("NUM_UPDATES", "300")))
    parser.add_argument("--python-bin", type=str, default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    check_required_files()
    run_name = f"ref_sweep_seed{args.seed}_u{args.num_updates}"
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
        "jobs": [],
    }

    processes: list[tuple[dict, subprocess.Popen, object]] = []
    for job in jobs:
        log_path = log_dir / f"{job['name']}.txt"
        cmd = [args.python_bin, job["script"], "--exp-name", job["name"], "--cuda-id", job["gpu"], *job["common"], *job["extra"]]
        job_meta = {"label": job["label"], "name": job["name"], "gpu": job["gpu"], "log": str(log_path), "cmd": cmd}
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
