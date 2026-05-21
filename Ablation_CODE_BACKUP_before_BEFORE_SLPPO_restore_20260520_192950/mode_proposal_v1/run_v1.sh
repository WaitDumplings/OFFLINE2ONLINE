#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/Maojie/Github2/EVRP-TW-D-B_Weekend"
PY="/home/npg/miniconda3/envs/maojie/bin/python"
BASE_CKPT="$ROOT/checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth"
LOG="$ROOT/LOGS/Codex_Res_mode_proposal_v1"
DATA="$ROOT/mode_proposal_v1/data"

mkdir -p "$LOG" "$DATA" "$ROOT/checkpoint/mode_proposal_v1"

"$PY" "$ROOT/mode_proposal_v1/solomon_to_pickle.py" \
  --solomon-dir "$ROOT/dataset/unanchored/Cus_50/solomon" \
  --output-pkl "$DATA/solomon_eval_1000.pkl"

"$PY" "$ROOT/mode_proposal_v1/policy_eval.py" \
  --checkpoint-path "$BASE_CKPT" \
  --eval-data-path "$ROOT/dataset/unanchored/Cus_50/buffer/pickle/evrptw_50C_12R.pkl" \
  --output-csv "$LOG/buffer_base_pomo50.csv" \
  --summary-json "$LOG/buffer_base_pomo50.summary.json" \
  --cuda-id 0 --n-traj 50 --eval-batch-size 64 --eval-steps 100

"$PY" "$ROOT/mode_proposal_v1/build_missing_mode_dataset.py" \
  --buffer-dir "$ROOT/dataset/unanchored/Cus_50/buffer" \
  --frontier-csv "$LOG/buffer_base_pomo50.csv" \
  --checkpoint-path "$BASE_CKPT" \
  --output-csv "$LOG/missing_mode_v1.csv" \
  --output-pkl "$DATA/missing_mode_v1_selected.pkl" \
  --summary-json "$LOG/missing_mode_v1.summary.json" \
  --cuda-id 0 --margin 0.03 --novelty-quantile 0.5 --replay-batch-size 64

"$PY" "$ROOT/mode_proposal_v1/train_offline_proposal.py" \
  --records-pkl "$DATA/missing_mode_v1_selected.pkl" \
  --base-checkpoint-path "$BASE_CKPT" \
  --output-checkpoint "$ROOT/checkpoint/mode_proposal_v1/offline_proposal_v1.pth" \
  --log-csv "$LOG/offline_proposal_v1_train.csv" \
  --summary-json "$LOG/offline_proposal_v1_train.summary.json" \
  --cuda-id 1 --epochs 8 --batch-size 24 --learning-rate 3e-5 --kl-coef 0.02

"$PY" "$ROOT/mode_proposal_v1/policy_eval.py" \
  --checkpoint-path "$BASE_CKPT" \
  --eval-data-path "$DATA/solomon_eval_1000.pkl" \
  --output-csv "$LOG/solomon_base50.csv" \
  --summary-json "$LOG/solomon_base50.summary.json" \
  --cuda-id 0 --n-traj 50 --eval-batch-size 64 --eval-steps 100

"$PY" "$ROOT/mode_proposal_v1/policy_eval.py" \
  --checkpoint-path "$BASE_CKPT" \
  --eval-data-path "$DATA/solomon_eval_1000.pkl" \
  --output-csv "$LOG/solomon_base40.csv" \
  --summary-json "$LOG/solomon_base40.summary.json" \
  --cuda-id 0 --n-traj 40 --eval-batch-size 64 --eval-steps 100

"$PY" "$ROOT/mode_proposal_v1/policy_eval.py" \
  --checkpoint-path "$BASE_CKPT" \
  --eval-data-path "$DATA/solomon_eval_1000.pkl" \
  --output-csv "$LOG/solomon_base32.csv" \
  --summary-json "$LOG/solomon_base32.summary.json" \
  --cuda-id 0 --n-traj 32 --eval-batch-size 64 --eval-steps 100

"$PY" "$ROOT/mode_proposal_v1/policy_eval.py" \
  --checkpoint-path "$BASE_CKPT" \
  --eval-data-path "$DATA/solomon_eval_1000.pkl" \
  --output-csv "$LOG/solomon_temp10_t1p5.csv" \
  --summary-json "$LOG/solomon_temp10_t1p5.summary.json" \
  --cuda-id 1 --n-traj 10 --eval-batch-size 128 --eval-steps 100 --temperature 1.5

"$PY" "$ROOT/mode_proposal_v1/policy_eval.py" \
  --checkpoint-path "$BASE_CKPT" \
  --eval-data-path "$DATA/solomon_eval_1000.pkl" \
  --output-csv "$LOG/solomon_temp8_t1p5.csv" \
  --summary-json "$LOG/solomon_temp8_t1p5.summary.json" \
  --cuda-id 1 --n-traj 8 --eval-batch-size 128 --eval-steps 100 --temperature 1.5

"$PY" "$ROOT/mode_proposal_v1/policy_eval.py" \
  --checkpoint-path "$ROOT/checkpoint/mode_proposal_v1/offline_proposal_v1.pth" \
  --eval-data-path "$DATA/solomon_eval_1000.pkl" \
  --output-csv "$LOG/solomon_off10_v1.csv" \
  --summary-json "$LOG/solomon_off10_v1.summary.json" \
  --cuda-id 1 --n-traj 10 --eval-batch-size 128 --eval-steps 100

"$PY" "$ROOT/mode_proposal_v1/summarize_portfolio.py" \
  --base50 "$LOG/solomon_base50.csv" \
  --base40 "$LOG/solomon_base40.csv" \
  --base32 "$LOG/solomon_base32.csv" \
  --temp10 "$LOG/solomon_temp10_t1p5.csv" \
  --temp8 "$LOG/solomon_temp8_t1p5.csv" \
  --off10 "$LOG/solomon_off10_v1.csv" \
  --output-json "$LOG/portfolio_v1.summary.json" \
  --output-dir "$LOG/portfolio_v1"
