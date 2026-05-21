#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_common(num_updates: int, save_dir: Path) -> list[str]:
    return [
        "--cuda", "true",
        "--torch-deterministic", "true",
        "--save-dir", str(save_dir),
        "--eval-data-path", str(ROOT / "dataset" / "unanchored" / "Cus_50" / "pickle" / "evrptw_50C_12R.pkl"),
        "--eval-freq", "10",
        "--eval-batch-size", "1000",
        "--num-updates", str(num_updates),
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


def incumbent_args() -> list[str]:
    return [
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
        "--reference-adv-gate-mode", "linear",
        "--reference-adv-gate-temp", "0.05",
        "--reference-adv-hard-threshold", "0.03",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SL-PPO advantage-split ablation on GPU0-3.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3007")))
    parser.add_argument("--num-updates", type=int, default=int(os.environ.get("NUM_UPDATES", "300")))
    parser.add_argument("--python", type=str, default=os.environ.get("PYTHON", sys.executable))
    args = parser.parse_args()

    tag = f"slppo_advsplit_seed{args.seed}_u{args.num_updates}"
    log_dir = ROOT / "LOGS" / tag
    img_dir = ROOT / "IMGS" / tag
    save_dir = ROOT / "checkpoint" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    common = build_common(args.num_updates, save_dir)
    inc = incumbent_args()
    ref_common = [
        "--reference-adv-coef", "0.10",
        "--reference-adv-rho", "0.05",
        "--reference-adv-clip", "2.0",
    ]
    group_common = [
        "--group-adv-coef", "0.30",
        "--group-adv-clip", "3.0",
    ]

    jobs = [
        {
            "name": f"gpu0_vanilla_ppo_2head_seed{args.seed}_u{args.num_updates}",
            "label": "Vanilla PPO 2-head, no offline archive guidance",
            "gpu": "0",
            "script": "train.py",
            "extra": ["--use-alns-teacher", "false", "--use-alns-bc", "false", "--use-alns-preference", "false"],
        },
        {
            "name": f"gpu1_ppo_archive_only_group_ref_seed{args.seed}_u{args.num_updates}",
            "label": "Step PPO actor advantage = group + soft-gated ref only; no GAE actor advantage",
            "gpu": "1",
            "script": "train_incumbent.py",
            "extra": inc + group_common + ref_common + [
                "--step-adv-mode", "archive_only",
                "--use-route-level-loss", "false",
                "--route-loss-coef", "0.0",
            ],
        },
        {
            "name": f"gpu2_ppo_gae_ref_seed{args.seed}_u{args.num_updates}",
            "label": "Step PPO actor advantage = GAE + soft-gated ref; no group advantage",
            "gpu": "2",
            "script": "train_incumbent.py",
            "extra": inc + ref_common + [
                "--step-adv-mode", "base_ref",
                "--group-adv-coef", "0.0",
                "--group-adv-clip", "3.0",
                "--use-route-level-loss", "false",
                "--route-loss-coef", "0.0",
            ],
        },
        {
            "name": f"gpu3_gae_step_group_ref_slppo_seed{args.seed}_u{args.num_updates}",
            "label": "Step PPO actor advantage = GAE only; SL-PPO route advantage = group + soft-gated ref",
            "gpu": "3",
            "script": "train_incumbent.py",
            "extra": inc + group_common + ref_common + [
                "--step-adv-mode", "base_only",
                "--use-route-level-loss", "true",
                "--route-loss-coef", "1.0",
                "--route-adv-source", "group_ref",
                "--route-clip-eps", "0.20",
                "--route-ratio-normalize", "mean_logprob",
                "--only-success-route-loss", "true",
                "--positive-only-route-loss", "false",
            ],
        },
    ]

    launched = []
    for job in jobs:
        log_path = log_dir / f"{job['name']}.txt"
        cmd = [
            args.python,
            job["script"],
            "--exp-name", job["name"],
            "--cuda-id", "0",
            "--seed", str(args.seed),
            *common,
            *job["extra"],
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = job["gpu"]
        fh = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
        payload = {
            "pid": proc.pid,
            "name": job["name"],
            "label": job["label"],
            "physical_gpu": job["gpu"],
            "visible_cuda_id": "0",
            "log": str(log_path),
            "cmd": cmd,
        }
        launched.append(payload)
        print(f"{job['name']} pid={proc.pid} gpu={job['gpu']} log={log_path}")

    (log_dir / "pids.json").write_text(json.dumps(launched, indent=2))
    (log_dir / "run_metadata.json").write_text(json.dumps({
        "seed": args.seed,
        "num_updates": args.num_updates,
        "root": str(ROOT),
        "tag": tag,
        "jobs": launched,
    }, indent=2))
    print(f"wrote {log_dir / 'pids.json'}")
    print(f"images should be saved under {img_dir}")


if __name__ == "__main__":
    main()
