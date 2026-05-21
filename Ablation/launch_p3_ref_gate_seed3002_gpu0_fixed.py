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
PID_PATH = LOG_DIR / "pid_gpu0_fixed.json"
NAME = "gpu0_p2_fixed_refcoef0p10_rho0p10_seed3002_u300"

CMD = [
    str(PY),
    "train_incumbent.py",
    "--exp-name", NAME,
    "--cuda-id", "0",
    "--seed", SEED,
    *COMMON,
    *INCUMBENT,
    "--group-adv-coef", "0.30",
    "--group-adv-clip", "3.0",
    "--reference-adv-coef", "0.10",
    "--reference-adv-rho", "0.10",
    "--reference-adv-clip", "2.0",
    "--reference-adv-alns-win-only", "true",
    "--reference-adv-gate-mode", "fixed",
    "--save-dir", str(SAVE_DIR),
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{NAME}.txt"
    fh = log_path.open("w")
    proc = subprocess.Popen(
        CMD,
        cwd=ROOT,
        stdout=fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    payload = {NAME: {"pid": proc.pid, "gpu": "0", "label": "P2 fixed ref baseline coef=0.10 rho=0.10", "log": str(log_path), "cmd": CMD}}
    PID_PATH.write_text(json.dumps(payload, indent=2))
    print(f"{NAME} pid={proc.pid} gpu=0 log={log_path}")
    print(f"wrote {PID_PATH}")


if __name__ == "__main__":
    main()
