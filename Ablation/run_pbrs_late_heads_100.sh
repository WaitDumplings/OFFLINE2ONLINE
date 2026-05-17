#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/Maojie/Github2/EVRP-TW-D-B_Weekend"
PY="${PY:-/home/npg/miniconda3/envs/maojie/bin/python}"
LOG_DIR="${ROOT}/LOGS/Codex_Res_pbrs_lateheads100_clean_snapshot_ne128_repair_progress_objprog_enc512_s75_ntraj50_nm128_u100"
EVAL_DATA="${ROOT}/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ARGS=(
  --num-updates 100
  --num-envs 128
  --n-traj 50
  --num-steps 75
  --num-minibatches 128
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
  --debug True
)

LATE_ADV_ARGS=(
  --use-decomposed-reward-adv True
  --adv-objective-weight 0.65
  --adv-progress-weight 0.20
  --adv-terminal-weight 0.10
  --adv-teacher-weight 0.0
  --use-adaptive-adv-weights False
)

launch() {
  local gpu="$1"
  local variant="$2"
  local extra_args="$3"
  local seed=2025
  local exp_name="ablation5_routeevent_${variant}_lateheads100_ne128_repair_progress_objprog_enc512_s75_ntraj50_nm128_gpu${gpu}_seed${seed}_u100"
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

# GPU0: clean no-PBRS reference, single critic.
launch 0 "pbrsN_none_singlecritic" "--pbrs-mode none --use-direct-progress-pbrs False --progress-pbrs-coef 0.0 --use-repair-progress-pbrs False --use-repair-fail-reward False --repair-progress-coef 0.0 --repair-progress-include-current True"

# GPU1: served-customer-ratio PBRS with 2 heads: objective + progress.
launch 1 "pbrsS_customer_ratio_2head" "--pbrs-mode served --use-direct-progress-pbrs False --progress-pbrs-coef 0.0 --use-repair-progress-pbrs False --use-repair-fail-reward False --repair-progress-coef 0.0 --repair-progress-include-current True --decomposed-reward-mode objective_progress ${LATE_ADV_ARGS[*]}"

# GPU2: repair heuristic PBRS with 2 heads: objective + progress.
launch 2 "pbrsH_heuristic_2head" "--pbrs-mode none --use-direct-progress-pbrs False --progress-pbrs-coef 0.0 --use-repair-progress-pbrs True --use-repair-fail-reward False --repair-progress-coef 1.0 --repair-progress-include-current True --decomposed-reward-mode objective_progress ${LATE_ADV_ARGS[*]}"

# GPU3: customer-ratio + repair heuristic PBRS with 3 heads: objective + progress + terminal.
launch 3 "pbrsA_all_3head" "--pbrs-mode served --use-direct-progress-pbrs False --progress-pbrs-coef 0.0 --use-repair-progress-pbrs True --use-repair-fail-reward False --repair-progress-coef 1.0 --repair-progress-include-current True --decomposed-reward-mode objective_progress_terminal ${LATE_ADV_ARGS[*]}"

echo "[done] launched PBRS late-heads 100-epoch sweep"
