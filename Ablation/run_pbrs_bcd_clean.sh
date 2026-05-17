#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/Maojie/Github2/EVRP-TW-D-B_Weekend"
PY="${PY:-/home/npg/miniconda3/envs/maojie/bin/python}"
LOG_DIR="${ROOT}/LOGS/Codex_Res_pbrs_bcd_clean_snapshot_memsafe_ne64_repair_progress_objprog_enc512_s75_ntraj50_nm64_u800"
EVAL_DATA="${ROOT}/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ARGS=(
  --num-updates 800
  --num-envs 64
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
  --train-config-log-freq 20
  --train-env-seed-by-update True
  --learning-rate 4e-5
  --critic-lr 3e-5
  --target-kl 0.01
  --n-encode-layers 2
  --max-route-events 16
  --reward-mode vanilla
  --progress-pbrs-beta 0.5
  --repair-fail-coef 1.0
  --repair-success-bonus 1.0
  --decomposed-reward-mode objective_progress_terminal_teacher
  --debug True
)

launch() {
  local gpu="$1"
  local variant="$2"
  local extra_args="$3"
  local seed=2025
  local exp_name="ablation5_routeevent_${variant}_cleanpbrs_memsafe_ne64_repair_progress_objprog_enc512_s75_ntraj50_nm64_gpu${gpu}_seed${seed}_u800"
  local log_path="${LOG_DIR}/result_gpu${gpu}_${exp_name}.txt"
  local save_dir="${ROOT}/checkpoint/${exp_name}"

  echo "[launch] gpu=${gpu} variant=${variant}"
  echo "[launch] log=${log_path}"
  echo "[launch] save_dir=${save_dir}"

  # shellcheck disable=SC2086
  nohup setsid "${PY}" -u -m evrptw_gen.benchmarks.DRL_Solver.DRL_train \
    --cuda-id "${gpu}" \
    --exp-name "${exp_name}" \
    --save-dir "${save_dir}" \
    --seed "${seed}" \
    "${COMMON_ARGS[@]}" \
    ${extra_args} \
    > "${log_path}" 2>&1 &

  echo "[launch] pid=$!"
}

# B: customer-only heuristic PBRS replaces customer-count PBRS.
launch 1 "pbrsB_customerheur" "--pbrs-mode none --use-direct-progress-pbrs False --progress-pbrs-coef 0.0 --use-repair-fail-reward True --repair-progress-coef 1.0 --repair-progress-include-current False"

# C: full repair heuristic PBRS replaces customer-count PBRS.
launch 2 "pbrsC_fullrepair" "--pbrs-mode none --use-direct-progress-pbrs False --progress-pbrs-coef 0.0 --use-repair-fail-reward True --repair-progress-coef 1.0 --repair-progress-include-current True"

# D: no PBRS, only distance objective plus terminal/constraint rewards.
launch 3 "pbrsD_nopbrs" "--pbrs-mode none --use-direct-progress-pbrs False --progress-pbrs-coef 0.0 --use-repair-fail-reward False --repair-progress-coef 0.0 --repair-progress-include-current True"

echo "[done] launched B/C/D"
