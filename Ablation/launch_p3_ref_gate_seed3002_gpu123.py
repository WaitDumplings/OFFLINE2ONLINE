#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from launch_pomo50_seed3001 import COMMON, INCUMBENT, PY, ROOT

SEED = "3002"
LOG_DIR = ROOT / "LOGS_NEW" / "P3_REF_GATE_seed3002"
IMG_DIR = ROOT / "IMGS_NEW" / "P3_REF_GATE_seed3002"
SAVE_DIR = ROOT / "checkpoint" / "P3_REF_GATE_seed3002"
PID_PATH = LOG_DIR / "pids_gpu123.json"

BASE_REF = [
    "--group-adv-coef", "0.30",
    "--group-adv-clip", "3.0",
    "--reference-adv-coef", "0.10",
    "--reference-adv-rho", "0.10",
    "--reference-adv-clip", "2.0",
    "--save-dir", str(SAVE_DIR),
]

JOBS = [
    {
        "name": "gpu1_p3_linear_all_refgate_seed3002_u300",
        "label": "P3 linear gate, all offline positive-regret cases",
        "script": "train_incumbent.py",
        "gpu": "1",
        "extra": INCUMBENT + BASE_REF + [
            "--reference-adv-alns-win-only", "false",
            "--reference-adv-gate-mode", "linear",
            "--reference-adv-gate-temp", "0.05",
            "--reference-adv-hard-threshold", "0.03",
        ],
    },
    {
        "name": "gpu2_p3_hard_all_refgate_seed3002_u300",
        "label": "P3 hard gate, all offline cases, regret > 3%",
        "script": "train_incumbent.py",
        "gpu": "2",
        "extra": INCUMBENT + BASE_REF + [
            "--reference-adv-alns-win-only", "false",
            "--reference-adv-gate-mode", "hard",
            "--reference-adv-gate-temp", "0.05",
            "--reference-adv-hard-threshold", "0.03",
        ],
    },
    {
        "name": "gpu3_p3_linear_stablealns_refgate_seed3002_u300",
        "label": "P3 linear gate, stable ALNS-win only",
        "script": "train_incumbent.py",
        "gpu": "3",
        "extra": INCUMBENT + BASE_REF + [
            "--reference-adv-alns-win-only", "true",
            "--reference-adv-gate-mode", "linear",
            "--reference-adv-gate-temp", "0.05",
            "--reference-adv-hard-threshold", "0.03",
        ],
    },
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    launched = {}
    for job in JOBS:
        log_path = LOG_DIR / f"{job['name']}.txt"
        cmd = [
            str(PY),
            job["script"],
            "--exp-name", job["name"],
            "--cuda-id", job["gpu"],
            "--seed", SEED,
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
        launched[job["name"]] = {
            "pid": proc.pid,
            "gpu": job["gpu"],
            "label": job["label"],
            "log": str(log_path),
            "cmd": cmd,
        }
        print(f"{job['name']} pid={proc.pid} gpu={job['gpu']} log={log_path}")
    PID_PATH.write_text(json.dumps(launched, indent=2))
    print(f"wrote {PID_PATH}")


if __name__ == "__main__":
    main()
