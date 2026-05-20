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
    entropy_gate = [
        "--use-decision-fork-gate", "true",
        "--decision-fork-gate-mode", "entropy",
        "--decision-fork-entropy-threshold", "0.60",
    ]
    feasible_gate = [
        "--use-decision-fork-gate", "true",
        "--decision-fork-gate-mode", "feasible_count",
        "--decision-fork-min-actions", "5",
        "--decision-fork-max-actions", "12",
    ]
    route_gate = [
        "--use-decision-fork-gate", "true",
        "--decision-fork-gate-mode", "route_structure",
    ]
    return [
        {
            "label": "Exp3 soft gate baseline",
            "name": f"gpu0_p2_exp3_softgate_full_seed{seed}_u{num_updates}",
            "gpu": "0",
            "note": "baseline: full-step broadcast of group/ref route-level signal",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu0_p2_exp3_softgate_full_seed{seed}_u{num_updates}")],
        },
        {
            "label": "Exp3 + entropy fork gate",
            "name": f"gpu1_p2_exp3_entropyfork_seed{seed}_u{num_updates}",
            "gpu": "1",
            "note": "apply group/ref signal only on high normalized-entropy steps, threshold=0.60",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *entropy_gate, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu1_p2_exp3_entropyfork_seed{seed}_u{num_updates}")],
        },
        {
            "label": "Exp3 + feasible-count fork gate",
            "name": f"gpu2_p2_exp3_feasiblefork_seed{seed}_u{num_updates}",
            "gpu": "2",
            "note": "apply group/ref signal only when feasible action count >= 5",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *feasible_gate, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu2_p2_exp3_feasiblefork_seed{seed}_u{num_updates}")],
        },
        {
            "label": "Exp3 + route-structure fork gate",
            "name": f"gpu3_p2_exp3_routefork_seed{seed}_u{num_updates}",
            "gpu": "3",
            "note": "apply group/ref signal at route-start, depot-boundary, RS, and continue-vs-return decisions",
            "common": common,
            "extra": [*INCUMBENT, *BASE_REF, *route_gate, *NO_ROUTE, "--save-dir", str(save_dir / f"gpu3_p2_exp3_routefork_seed{seed}_u{num_updates}")],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch RGAE Priority-2 decision-fork gate experiments on GPU0-3.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3005")))
    parser.add_argument("--num-updates", type=int, default=int(os.environ.get("NUM_UPDATES", "300")))
    parser.add_argument("--python-bin", type=str, default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    check_required_files()
    run_name = f"rgae_p2_fork_seed{args.seed}_u{args.num_updates}"
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
            "name": "RGAE Priority-2 decision-fork gate",
            "base": "Exp3 soft gate",
            "group_adv_coef": 0.30,
            "reference_adv_coef": 0.10,
            "reference_adv_rho": 0.10,
            "reference_adv_gate_mode": "linear",
            "soft_gate_eta": 0.05,
            "n_traj": 50,
            "test_agent": 8,
            "fork_variants": {
                "gpu0": "none/full broadcast",
                "gpu1": "entropy threshold 0.60",
                "gpu2": "feasible count >= 5",
                "gpu3": "route structure decisions",
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
            "--cuda-id", job["gpu"],
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
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
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
