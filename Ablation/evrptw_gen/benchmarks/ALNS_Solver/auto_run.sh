nohup python main.py \
  --instance_path ../../../dataset/unanchored/Cus_50/buffer/solomon \
  --save_log_path ./logs_Cus50_init200 \
  --progress_path ../../../dataset/unanchored/Cus_50/buffer/progress/buffer_progress.pkl \
  --delta_iters 200 \
  > result_init200_64bs.txt 2>&1 &