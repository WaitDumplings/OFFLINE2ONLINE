#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export SEED="${SEED:-3008}"
export NUM_UPDATES="${NUM_UPDATES:-2000}"
PYTHON_BIN="${PYTHON:-python}"
TAG="slppo_ablation_test_seed${SEED}_u${NUM_UPDATES}"
LOG_DIR="$ROOT/LOGS/$TAG"
IMG_DIR="$ROOT/IMGS/$TAG"
CKPT_DIR="$ROOT/checkpoint/$TAG"
LAUNCH_LOG="$LOG_DIR/launch_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$IMG_DIR" "$CKPT_DIR"

echo "ROOT=$ROOT"
echo "PYTHON=$PYTHON_BIN"
echo "SEED=$SEED"
echo "NUM_UPDATES=$NUM_UPDATES"
echo "LOG_DIR=$LOG_DIR"
echo "IMG_DIR=$IMG_DIR"
echo "CKPT_DIR=$CKPT_DIR"
echo "launch log=$LAUNCH_LOG"

COMMON=(
  --cuda true
  --torch-deterministic true
  --save-dir "$CKPT_DIR"
  --eval-data-path "$ROOT/dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl"
  --eval-freq 10
  --eval-batch-size 1000
  --num-updates "$NUM_UPDATES"
  --num-envs 128
  --num-steps 75
  --n-traj 50
  --test-agent 8
  --num-minibatches 128
  --accum-steps 16
  --eval-decode-mode sampling
  --train-config-schedule batch_cycle
  --train-config-stratify-keys instance_type,time_window_policy
  --train-config-fixed-overrides service_time_policy=cargoweight,cluster_number_policy=random
  --train-config-use-online-counter true
  --train-env-seed-by-update true
  --use-direct-progress-pbrs true
  --progress-pbrs-coef 2.0
  --progress-pbrs-beta 0.5
  --use-repair-fail-reward true
  --repair-fail-coef 1.0
  --repair-progress-coef 1.0
  --repair-success-bonus 1.0
  --use-decomposed-reward-adv true
  --decomposed-reward-mode objective_progress
  --adv-objective-weight 0.5
  --adv-progress-weight 0.5
  --debug true
  --debug-test true
)

INCUMBENT=(
  --adv-weight-schedule fixed
  --alns-buffer-dir "$ROOT/dataset/unanchored/Cus_50/buffer"
  --incumbent-ratio 0.20
  --buffer-normal-frac 1.0
  --buffer-temp-frac 0.0
  --buffer-prefix-frac 0.0
  --online-normal-frac 1.0
  --online-temp-frac 0.0
  --temperature-sampling 1.0
  --use-regret-aware-sampler true
  --use-pomo50-regret-state true
  --pomo50-stable-beat-rate 0.20
  --pomo50-low-beat-rate 0.05
  --pomo50-regret-margin-rel 0.01
  --buffer-regret-relative true
  --buffer-regret-kappa 25.0
  --buffer-regret-margin 0.0
  --buffer-unknown-weight 1.0
  --buffer-uncertain-weight 0.5
  --buffer-ppo-win-weight 0.05
  --buffer-lucky-ppo-weight 0.35
  --buffer-alns-win-base-weight 1.0
  --regret-recompute-freq 10
  --regret-probe-size 256
  --use-route-preference false
  --use-selective-bc false
  --use-self-generated-buffer false
  --cmp-adv-coef 0.0
  --adv-diag true
  --grad-cos-diag true
  --grad-cos-freq 10
  --reference-adv-alns-win-only true
  --reference-adv-gate-mode linear
  --reference-adv-gate-temp 0.05
  --reference-adv-hard-threshold 0.03
  --step-adv-mode base_only
  --use-route-level-loss true
  --route-loss-coef 1.0
  --route-clip-eps 0.20
  --route-ratio-normalize mean_logprob
  --only-success-route-loss true
  --positive-only-route-loss false
)

launch() {
  local gpu="$1"
  local name="$2"
  local script="$3"
  shift 3
  local log_path="$LOG_DIR/${name}.txt"
  echo "launch $name on GPU$gpu -> $log_path"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON_BIN" "$script" \
      --exp-name "$name" \
      --cuda-id 0 \
      --seed "$SEED" \
      "${COMMON[@]}" \
      "$@"
  ) > "$log_path" 2>&1 &
  local pid=$!
  echo "${pid}|${gpu}|${name}|${log_path}" >> "$LOG_DIR/pids.tsv"
  echo "$name pid=$pid gpu=$gpu log=$log_path"
}

: > "$LOG_DIR/pids.tsv"

{
  echo "[Ablation Test] seed=$SEED num_updates=$NUM_UPDATES"
  echo "GPU0: Vanilla PPO 2-head"
  echo "GPU1: step PPO=GAE only, SL-PPO route advantage=group only"
  echo "GPU2: step PPO=GAE only, SL-PPO route advantage=reference only"
  echo "GPU3: step PPO=GAE only, SL-PPO route advantage=group+reference"
  echo "Offline ratio for GPU1-3: incumbent-ratio=0.20 => expected buffer_normal=26, online_normal=102 when num_envs=128"
} | tee "$LAUNCH_LOG"

launch 0 "gpu0_vanilla_ppo_2head_seed${SEED}_u${NUM_UPDATES}" train.py \
  --use-alns-teacher false \
  --use-alns-bc false \
  --use-alns-preference false

launch 1 "gpu1_gae_slppo_group_seed${SEED}_u${NUM_UPDATES}" train_incumbent.py \
  "${INCUMBENT[@]}" \
  --group-adv-coef 0.30 \
  --group-adv-clip 3.0 \
  --reference-adv-coef 0.0 \
  --reference-adv-rho 0.05 \
  --reference-adv-clip 2.0 \
  --route-adv-source group

launch 2 "gpu2_gae_slppo_ref_seed${SEED}_u${NUM_UPDATES}" train_incumbent.py \
  "${INCUMBENT[@]}" \
  --group-adv-coef 0.0 \
  --group-adv-clip 3.0 \
  --reference-adv-coef 0.10 \
  --reference-adv-rho 0.05 \
  --reference-adv-clip 2.0 \
  --route-adv-source ref

launch 3 "gpu3_gae_slppo_group_ref_seed${SEED}_u${NUM_UPDATES}" train_incumbent.py \
  "${INCUMBENT[@]}" \
  --group-adv-coef 0.30 \
  --group-adv-clip 3.0 \
  --reference-adv-coef 0.10 \
  --reference-adv-rho 0.05 \
  --reference-adv-clip 2.0 \
  --route-adv-source group_ref

python - <<PY
from pathlib import Path
import json
log_dir = Path("$LOG_DIR")
rows = []
for line in (log_dir / "pids.tsv").read_text().splitlines():
    pid, gpu, name, log = line.split("|", 3)
    rows.append({"pid": int(pid), "gpu": gpu, "name": name, "log": log})
(log_dir / "pids.json").write_text(json.dumps(rows, indent=2))
(log_dir / "run_metadata.json").write_text(json.dumps({
    "seed": int("$SEED"),
    "num_updates": int("$NUM_UPDATES"),
    "tag": "$TAG",
    "offline_ratio": 0.20,
    "expected_groups": "buffer_normal=26,online_normal=102 for GPU1-3",
    "jobs": rows,
}, indent=2))
print(f"wrote {log_dir / 'pids.json'}")
print(f"wrote {log_dir / 'run_metadata.json'}")
PY

echo "All jobs launched. Monitor with: tail -f $LOG_DIR/*.txt"
