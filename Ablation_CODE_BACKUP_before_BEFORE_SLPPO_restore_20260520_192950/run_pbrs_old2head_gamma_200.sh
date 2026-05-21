#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/Maojie/Github2/EVRP-TW-D-B_Weekend"
PY="${PY:-/home/npg/miniconda3/envs/maojie/bin/python}"
LOG_DIR="${ROOT}/LOGS/Codex_Res_pbrs_old2head_gamma200_ne128_mb64_repair_progress_objprog_enc512_s75_ntraj50_u200"
EVAL_DATA="${ROOT}/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ARGS=(
  --num-updates 200
  --num-envs 128
  --n-traj 50
  --num-steps 75
  --num-minibatches 64
  --update-epochs 5
  --accum-steps 8
  --test-agent 8
  --eval-decode-mode sampling
  --eval-greedy-too False
  --eval-data-path "${EVAL_DATA}"
  --eval-freq 20
  --eval-batch-size 1024
  --train-cus-num 50
  --train-cs-num 12
  --train-config-schedule batch_cycle
  --train-config-stratify-keys instance_type,time_window_policy
  --train-config-fixed-overrides service_time_policy=cargoweight,cluster_number_policy=random
  --train-config-use-online-counter True
  --train-config-log True
  --train-config-log-freq 10
  --train-env-seed-by-update True
  --learning-rate 4e-5
  --critic-lr 3e-5
  --gamma 0.99
  --target-kl 0.01
  --n-encode-layers 2
  --max-route-events 16
  --reward-mode vanilla
  --pbrs-mode none
  --use-direct-progress-pbrs True
  --use-repair-progress-pbrs False
  --use-repair-fail-reward True
  --repair-progress-coef 1.0
  --repair-fail-coef 1.0
  --repair-success-bonus 1.0
  --repair-progress-include-current True
  --decomposed-reward-mode objective_progress
  --use-decomposed-reward-adv True
  --adv-objective-weight 0.5
  --adv-progress-weight 0.5
  --adv-terminal-weight 0.0
  --adv-teacher-weight 0.0
  --use-adaptive-adv-weights False
  --debug True
)

launch() {
  local gpu="$1"
  local variant="$2"
  local progress_coef="$3"
  local beta="$4"
  local seed=2025
  local exp_name="ablation5_routeevent_${variant}_old2head_gamma200_ne128_mb64_repair_progress_objprog_enc512_s75_ntraj50_gpu${gpu}_seed${seed}_u200"
  local log_path="${LOG_DIR}/result_gpu${gpu}_${exp_name}.txt"
  local save_dir="${ROOT}/checkpoint/${exp_name}"

  echo "[launch] gpu=${gpu} variant=${variant} progress_coef=${progress_coef} beta=${beta}"
  echo "[launch] log=${log_path}"
  echo "[launch] save_dir=${save_dir}"

  nohup setsid "${PY}" -u -m evrptw_gen.benchmarks.DRL_Solver.DRL_train \
    --cuda-id "${gpu}" \
    --exp-name "${exp_name}" \
    --save-dir "${save_dir}" \
    --seed "${seed}" \
    "${COMMON_ARGS[@]}" \
    --progress-pbrs-coef "${progress_coef}" \
    --progress-pbrs-beta "${beta}" \
    > "${log_path}" 2>&1 &

  echo "[launch] pid=$!"
}

launch 0 "pbrsG_c2p0_b0p75" 2.0 0.75
launch 1 "pbrsG_c2p0_b1p00" 2.0 1.00
launch 2 "pbrsG_c1p0_b0p75" 1.0 0.75
launch 3 "pbrsG_c1p0_b1p00" 1.0 1.00

echo "[done] launched gamma-corrected old-2head PBRS 200-epoch sweep"
