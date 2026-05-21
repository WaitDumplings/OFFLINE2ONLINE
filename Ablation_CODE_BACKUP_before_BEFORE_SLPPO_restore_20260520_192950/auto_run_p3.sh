#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export SEED="${SEED:-3006}"
export NUM_UPDATES="${NUM_UPDATES:-300}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CONDA_ENV="${CONDA_DEFAULT_ENV:-unknown}"

mkdir -p LOGS IMGS checkpoint
LAUNCH_LOG="LOGS/auto_run_p3_seed${SEED}_u${NUM_UPDATES}_$(date +%Y%m%d_%H%M%S).log"

echo "ROOT=$ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CONDA_DEFAULT_ENV=$CONDA_ENV"
echo "SEED=$SEED"
echo "NUM_UPDATES=$NUM_UPDATES"
echo "launch log=$LAUNCH_LOG"

if [[ "$CONDA_ENV" == "base" && "${ALLOW_BASE_PYTHON:-0}" != "1" ]]; then
  echo "ERROR: current conda env is base. Activate the training env first, or pass PYTHON_BIN explicitly."
  echo "Example: conda activate maojie"
  echo "Example: PYTHON_BIN=/home/npg0/anaconda3/envs/maojie/bin/python SEED=$SEED NUM_UPDATES=$NUM_UPDATES ./auto_run_p3.sh"
  exit 2
fi

"$PYTHON_BIN" - <<'PYCHECK'
import importlib.util, sys
missing = [m for m in ("torch", "numpy", "scipy", "gym") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing Python packages in selected PYTHON_BIN: " + ", ".join(missing))
print("python_check=ok", sys.executable)
PYCHECK

nohup "$PYTHON_BIN" run_p3.py --seed "$SEED" --num-updates "$NUM_UPDATES" --python-bin "$PYTHON_BIN" 2>&1 | tee "$LAUNCH_LOG"
