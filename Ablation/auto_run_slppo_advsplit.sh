#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export SEED="${SEED:-3007}"
export NUM_UPDATES="${NUM_UPDATES:-300}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p LOGS IMGS checkpoint
LAUNCH_LOG="LOGS/auto_run_slppo_advsplit_seed${SEED}_u${NUM_UPDATES}_$(date +%Y%m%d_%H%M%S).log"

echo "ROOT=$ROOT"
echo "PYTHON=$PYTHON_BIN"
echo "SEED=$SEED"
echo "NUM_UPDATES=$NUM_UPDATES"
echo "launch log=$LAUNCH_LOG"

"$PYTHON_BIN" run_slppo_advsplit.py --seed "$SEED" --num-updates "$NUM_UPDATES" 2>&1 | tee "$LAUNCH_LOG"
