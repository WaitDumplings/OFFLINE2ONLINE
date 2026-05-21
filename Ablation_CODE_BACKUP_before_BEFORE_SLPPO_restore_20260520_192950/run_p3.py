#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_rgae_v2 import ROOT, COMMON, INCUMBENT, BASE_REF, NO_ROUTE, check_required_files


def build_jobs(seed: int, num_updates: int, save_dir: Path) -> list[dict]:
    common = ["--seed", str(seed), "--num-updates", str(num_updates), *COMMON]
    sym_clip = [
        "--ppo-clip-mode", "symmetric",
        "--ppo-clip-low", "0.20",
        "--ppo-clip-high", "0.20",
    ]
    asym30 = [
        "--ppo-clip-mode", "asym",
        "--ppo-clip-low", "0.20",
        "--ppo-clip-high", "0.30",
    ]
    asym40 = [
        "--ppo-clip-mode", "asym",
        "--ppo-clip-low", "0.20",
        "--ppo-clip-high", "0.40",
    ]
    arch_asym40 = [
        "--ppo-clip-mode", "archive_asym",
        "--archive-clip-low", "0.20",
        "--archive-clip-high", "0.40",
    ]
    return [
        {
            "label": "Exp3 soft gate symmetric clip baseline",
            "name": f"gpu0_p3_exp3_softgate_sym_seed{seed}_u{num_updates}",
            "gpu": "0",
            "note": "baseline: Exp3 soft gate with standard symmetric PPO clipping 0.20/0.20",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *sym_clip, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu0_p3_exp3_softgate_sym_seed{seed}_u{num_updates}")],
        },
        {
            "label": "Exp3 + AsymClip 0.30",
            "name": f"gpu1_p3_exp3_asym30_seed{seed}_u{num_updates}",
            "gpu": "1",
            "note": "whole actor advantage uses asymmetric PPO clipping 0.20/0.30",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *asym30, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu1_p3_exp3_asym30_seed{seed}_u{num_updates}")],
        },
        {
            "label": "Exp3 + AsymClip 0.40",
            "name": f"gpu2_p3_exp3_asym40_seed{seed}_u{num_updates}",
            "gpu": "2",
            "note": "whole actor advantage uses asymmetric PPO clipping 0.20/0.40",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *asym40, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu2_p3_exp3_asym40_seed{seed}_u{num_updates}")],
        },
        {
            "label": "Exp3 + archive-only AsymClip 0.40",
            "name": f"gpu3_p3_exp3_archive_asym40_seed{seed}_u{num_updates}",
            "gpu": "3",
            "note": "GAE branch keeps symmetric 0.20/0.20; archive group/ref branch uses 0.20/0.40",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *arch_asym40, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu3_p3_exp3_archive_asym40_seed{seed}_u{num_updates}")],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch RGAE Priority-3 asymmetric clipping experiments on GPU0-3.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3006")))
    parser.add_argument("--num-updates", type=int, default=int(os.environ.get("NUM_UPDATES", "300")))
    parser.add_argument("--python-bin", type=str, default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    check_required_files()
    run_name = f"rgae_p3_asymclip_seed{args.seed}_u{args.num_updates}"
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
        "method": {
            "name": "RGAE Priority-3 asymmetric clipping",
            "base": "Exp3 soft gate",
            "group_adv_coef": 0.30,
            "reference_adv_coef": 0.10,
            "reference_adv_rho": 0.10,
            "reference_adv_gate_mode": "linear",
            "soft_gate_eta": 0.05,
            "n_traj": 50,
            "test_agent": 8,
            "clip_variants": {
                "gpu0": "symmetric 0.20/0.20",
                "gpu1": "whole actor asymmetric 0.20/0.30",
                "gpu2": "whole actor asymmetric 0.20/0.40",
                "gpu3": "archive-only asymmetric 0.20/0.40",
            },
        },
        "jobs": [],
    }

    processes = []
    for job in jobs:
        log_path = log_dir / f"{job['name']}.txt"
        cmd = [
            args.python_bin,
            "train_incumbent.py",
            "--exp-name", job["name"],
            # CUDA_VISIBLE_DEVICES below maps the selected physical GPU to local cuda:0.
            "--cuda-id", "0",
            *job["common"],
            *job["extra"],
        ]
        job_meta = {
            "label": job["label"],
            "name": job["name"],
            "gpu": job["gpu"],
            "note": job["note"],
            "log": str(log_path),
            "cmd": cmd,
        }
        metadata["jobs"].append(job_meta)
        if args.dry_run:
            print(f"[dry-run] {job['name']} GPU{job['gpu']}: {' '.join(cmd)}")
            continue
        fh = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = job["gpu"]
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        processes.append((proc, fh, job_meta))
        print(f"{job['name']} pid={proc.pid} gpu={job['gpu']} log={log_path}")
        time.sleep(2)

    metadata_path = log_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"wrote {metadata_path}")
    if not args.dry_run:
        pids = [{"pid": p.pid, **m} for p, _, m in processes]
        (log_dir / "pids.json").write_text(json.dumps(pids, indent=2))
        print(f"wrote {log_dir / 'pids.json'}")
        for _, fh, _ in processes:
            fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
