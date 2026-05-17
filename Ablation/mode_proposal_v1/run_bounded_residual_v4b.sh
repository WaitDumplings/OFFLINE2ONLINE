#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/Maojie/Github2/EVRP-TW-D-B_Weekend"
PY="/home/npg/miniconda3/envs/maojie/bin/python"
BASE_CKPT="$ROOT/checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth"
LOG="$ROOT/LOGS/Codex_Res_mode_proposal_v1"
DATA="$ROOT/mode_proposal_v1/data"

mkdir -p "$LOG" "$DATA" "$ROOT/checkpoint/mode_proposal_v1"

"$PY" "$ROOT/mode_proposal_v1/train_bounded_residual_proposal.py" \
  --records-pkl "$DATA/missing_mode_v1_selected.pkl" \
  --base-checkpoint-path "$BASE_CKPT" \
  --output-head "$ROOT/checkpoint/mode_proposal_v1/bounded_residual_head_v4b.pth" \
  --log-csv "$LOG/bounded_residual_head_v4b_train.csv" \
  --summary-json "$LOG/bounded_residual_head_v4b_train.summary.json" \
  --cuda-id 1 \
  --epochs 4 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --kl-coef 0.2 \
  --res-coef 0.02 \
  --max-residual 0.75 \
  --base-prob-threshold 0.15 \
  --rank-threshold 10

"$PY" "$ROOT/mode_proposal_v1/eval_bounded_residual_proposal.py" \
  --base-checkpoint-path "$BASE_CKPT" \
  --head-path "$ROOT/checkpoint/mode_proposal_v1/bounded_residual_head_v4b.pth" \
  --eval-data-path "$DATA/solomon_eval_1000.pkl" \
  --output-csv "$LOG/solomon_bounded_residual10_v4b.csv" \
  --summary-json "$LOG/solomon_bounded_residual10_v4b.summary.json" \
  --cuda-id 1 \
  --n-traj 10 \
  --eval-batch-size 128 \
  --eval-steps 100

"$PY" "$ROOT/mode_proposal_v1/summarize_portfolio.py" \
  --base50 "$LOG/solomon_base50.csv" \
  --base40 "$LOG/solomon_base40.csv" \
  --base32 "$LOG/solomon_base32.csv" \
  --temp10 "$LOG/solomon_temp10_t1p5.csv" \
  --temp8 "$LOG/solomon_temp8_t1p5.csv" \
  --off10 "$LOG/solomon_bounded_residual10_v4b.csv" \
  --output-json "$LOG/portfolio_v4b_bounded.summary.json" \
  --output-dir "$LOG/portfolio_v4b_bounded"
