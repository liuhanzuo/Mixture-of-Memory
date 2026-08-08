#!/bin/bash
# Launch keep10+fresh2 resume on .21 (8xL20A, wzc1 disk)
# This script waits for dolmino transfer to complete, then starts training
# eff_batch = batch_size(8) * grad_accum(2) * world_size(8) = 128
# USER INSTRUCTION 2026-08-08: B200用于resume, .21 = keep10 resume (not SparseForge)

PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PYTHON=/opt/conda/envs/torch-base/bin/python

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

cd $PROJECT_ROOT

LOG_FILE=$PROJECT_ROOT/logs/olmo2_7B_keep10fresh2_resume200k_21.log

echo "=== keep10+fresh2 resume on .21 started at $(date) ===" | tee $LOG_FILE

# Wait for dolmino transfer to complete (check every 60s, max 4h)
DATA_PATH=/dev/shm/dolmino_now15b_zwfy6.npy
EXPECTED_ROWS=15491607
EXPECTED_SIZE=126907244672  # 118 GB = 15491607 * 2048 * 4 bytes

echo "Waiting for dolmino transfer to complete at $DATA_PATH..." | tee -a $LOG_FILE

MAX_WAIT=14400  # 4 hours
WAITED=0
while true; do
  if [ -f "$DATA_PATH" ]; then
    CURRENT_SIZE=$(stat -c %s "$DATA_PATH" 2>/dev/null || echo 0)
    if [ "$CURRENT_SIZE" = "$EXPECTED_SIZE" ]; then
      echo "Data file complete: $DATA_PATH size=$CURRENT_SIZE" | tee -a $LOG_FILE
      break
    else
      echo "$(date): Data file partial: $CURRENT_SIZE / $EXPECTED_SIZE bytes, waiting..." | tee -a $LOG_FILE
    fi
  else
    echo "$(date): Data file not yet present, waiting..." | tee -a $LOG_FILE
  fi
  sleep 120
  WAITED=$((WAITED + 120))
  if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Timeout waiting for dolmino transfer" | tee -a $LOG_FILE
    exit 1
  fi
done

# Verify rows
DATA_ROWS=$($PYTHON -c "import numpy as np; d=np.load('$DATA_PATH', mmap_mode='r'); print(d.shape[0])" 2>/dev/null)
if [ "$DATA_ROWS" != "$EXPECTED_ROWS" ]; then
  echo "ERROR: dataset rows=$DATA_ROWS expected $EXPECTED_ROWS" | tee -a $LOG_FILE
  exit 1
fi
echo "Data verified: $DATA_PATH rows=$DATA_ROWS" | tee -a $LOG_FILE

# Verify ckpt
CKPT=$PROJECT_ROOT/outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt
if [ ! -f "$CKPT" ]; then
  echo "ERROR: $CKPT not found" | tee -a $LOG_FILE
  exit 1
fi
echo "Ckpt verified: $CKPT" | tee -a $LOG_FILE

echo "Starting torchrun at $(date)..." | tee -a $LOG_FILE

torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --resume_from $CKPT \
  --keep_front_layers 10 \
  --n_fresh_layers 2 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --seq_len 2048 \
  --lr 2e-5 \
  --min_lr 2e-6 \
  --lr_inherited 2e-5 \
  --max_steps 200000 \
  --warmup_steps 150 \
  --weight_decay 0.1 \
  --gradient_checkpointing 1 \
  --save_every 500 \
  --milestone_every 5000 \
  --keep_last_n 3 \
  --keep_milestones 8 \
  --keep_steps 83500,121000,124000,150000,175000,200000 \
  --data_path $DATA_PATH \
  --output_dir $PROJECT_ROOT/outputs/olmo2_probe2_7B_keep10fresh2 \
  --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  2>&1 | tee -a $LOG_FILE
