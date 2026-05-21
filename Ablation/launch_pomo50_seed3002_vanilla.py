#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from launch_pomo50_seed3001 import COMMON, PY, ROOT


SEED = "3002"
LOG_DIR = ROOT / "LOGS_NEW" / "RGAE_POMO50_300_seed3002"
IMG_DIR = ROOT / "IMGS_NEW" / "RGAE_POMO50_300_seed3002"
PID_PATH = LOG_DIR / "pid_vanilla.json"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    name = "gpu0_vanilla_ppo_2head_seed3002_u300"
    log_path = LOG_DIR / f"{name}.txt"
    cmd = [
        str(PY),
        "train.py",
        "--exp-name", name,
        "--cuda-id", "0",
        "--seed", SEED,
        *COMMON,
        "--use-alns-teacher", "false",
        "--use-alns-bc", "false",
        "--use-alns-preference", "false",
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
    info = {name: {"pid": proc.pid, "gpu": "0", "log": str(log_path), "cmd": cmd}}
    PID_PATH.write_text(json.dumps(info, indent=2))
    print(f"{name} pid={proc.pid} gpu=0 log={log_path}")
    print(f"wrote {PID_PATH}")


if __name__ == "__main__":
    main()
