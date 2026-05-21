#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from launch_pomo50_seed3001 import COMMON, INCUMBENT, PY, ROOT


SEED = "3002"
LOG_DIR = ROOT / "LOGS_NEW" / "RGAE_POMO50_300_seed3002"
IMG_DIR = ROOT / "IMGS_NEW" / "RGAE_POMO50_300_seed3002"
PID_PATH = LOG_DIR / "pids_exp123.json"


JOBS = [
    {
        "name": "gpu1_pomo50_exp1_sampler_seed3002_u300",
        "script": "train_incumbent.py",
        "gpu": "1",
        "extra": INCUMBENT + ["--group-adv-coef", "0.0", "--reference-adv-coef", "0.0"],
    },
    {
        "name": "gpu2_pomo50_exp2_group_seed3002_u300",
        "script": "train_incumbent.py",
        "gpu": "2",
        "extra": INCUMBENT + ["--group-adv-coef", "0.30", "--group-adv-clip", "3.0", "--reference-adv-coef", "0.0"],
    },
    {
        "name": "gpu3_pomo50_exp3_group_ref_seed3002_u300",
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
        launched[job["name"]] = {"pid": proc.pid, "gpu": job["gpu"], "log": str(log_path), "cmd": cmd}
        print(f"{job['name']} pid={proc.pid} gpu={job['gpu']} log={log_path}")
    PID_PATH.write_text(json.dumps(launched, indent=2))
    print(f"wrote {PID_PATH}")


if __name__ == "__main__":
    main()
