#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/data/Maojie/Github2/EVRP-TW-D-B_Weekend")
PY = Path("/home/npg/miniconda3/envs/maojie/bin/python")
LOG_DIR = ROOT / "LOGS_NEW" / "RGAE_POMO50_300_seed3001"
IMG_DIR = ROOT / "IMGS_NEW" / "RGAE_POMO50_300_seed3001"
PID_PATH = LOG_DIR / "pids.json"


COMMON = [
    "--cuda", "true",
    "--torch-deterministic", "true",
    "--save-dir", str(ROOT / "checkpoint" / "RGAE_POMO50"),
    "--eval-data-path", str(ROOT / "dataset" / "unanchored" / "Cus_50" / "pickle" / "evrptw_50C_12R.pkl"),
    "--eval-freq", "10",
    "--eval-batch-size", "1000",
    "--num-updates", "300",
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
]


JOBS = [
    {
        "name": "gpu0_vanilla_ppo_2head_seed3001_u300",
        "script": "train.py",
        "gpu": "0",
        "extra": ["--use-alns-teacher", "false", "--use-alns-bc", "false", "--use-alns-preference", "false"],
    },
    {
        "name": "gpu1_pomo50_exp1_sampler_seed3001_u300",
        "script": "train_incumbent.py",
        "gpu": "1",
        "extra": INCUMBENT + ["--group-adv-coef", "0.0", "--reference-adv-coef", "0.0"],
    },
    {
        "name": "gpu2_pomo50_exp2_group_seed3001_u300",
        "script": "train_incumbent.py",
        "gpu": "2",
        "extra": INCUMBENT + ["--group-adv-coef", "0.30", "--group-adv-clip", "3.0", "--reference-adv-coef", "0.0"],
    },
    {
        "name": "gpu3_pomo50_exp3_group_ref_seed3001_u300",
        "script": "train_incumbent.py",
        "gpu": "3",
        "extra": INCUMBENT + [
            "--group-adv-coef", "0.30",
            "--group-adv-clip", "3.0",
            "--reference-adv-coef", "0.10",
            "--reference-adv-rho", "0.05",
            "--reference-adv-clip", "2.0",
        ],
    },
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    launched = {}
    for job in JOBS:
        log_path = LOG_DIR / f"{job['name']}.txt"
        cmd = [
            str(PY),
            job["script"],
            "--exp-name", job["name"],
            "--cuda-id", job["gpu"],
            "--seed", "3001",
            *COMMON,
            *job["extra"],
        ]
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
        launched[job["name"]] = {"pid": proc.pid, "gpu": job["gpu"], "log": str(log_path), "cmd": cmd}
        print(f"{job['name']} pid={proc.pid} gpu={job['gpu']} log={log_path}")
    PID_PATH.write_text(json.dumps(launched, indent=2))
    print(f"wrote {PID_PATH}")


if __name__ == "__main__":
    main()
