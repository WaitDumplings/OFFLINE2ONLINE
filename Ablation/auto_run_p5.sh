#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export SEED="${SEED:-3003}"
export NUM_UPDATES="${NUM_UPDATES:-300}"

mkdir -p LOGS IMGS checkpoint
LAUNCH_LOG="LOGS/auto_run_p5_seed${SEED}_u${NUM_UPDATES}_$(date +%Y%m%d_%H%M%S).log"

echo "ROOT=$ROOT"
echo "PYTHON=$(command -v python)"
echo "SEED=$SEED"
echo "NUM_UPDATES=$NUM_UPDATES"
echo "launch log=$LAUNCH_LOG"

nohup python run_p5.py --seed "$SEED" --num-updates "$NUM_UPDATES" 2>&1 | tee "$LAUNCH_LOG"
