#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-/home/npg/miniconda3/envs/maojie/bin/python}"
export SEED="${SEED:-3003}"
export NUM_UPDATES="${NUM_UPDATES:-300}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  echo "Set it explicitly, for example: PYTHON_BIN=/path/to/python SEED=3003 ./all_run.sh" >&2
  exit 1
fi

mkdir -p LOGS IMGS checkpoint
LAUNCH_LOG="LOGS/all_run_seed${SEED}_u${NUM_UPDATES}_$(date +%Y%m%d_%H%M%S).log"

echo "ROOT=$ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SEED=$SEED"
echo "NUM_UPDATES=$NUM_UPDATES"
echo "launch log=$LAUNCH_LOG"

"$PYTHON_BIN" run_ablation.py --seed "$SEED" --num-updates "$NUM_UPDATES" --python-bin "$PYTHON_BIN" 2>&1 | tee "$LAUNCH_LOG"
