#!/bin/bash
# Launch keep12+fresh2 resume on .21 (8xL20A, wzc1 disk)
# Uses zwfy6 15.5M corpus (transferred to /dev/shm/dolmino_now15b_zwfy6.npy)
# Resumes from step124000.pt (transferred from zwfy6 via PART 1)
# eff_batch = batch_size(8) * grad_accum(2) * world_size(8) = 128

set -e

PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PYTHON=/opt/conda/envs/torch-base/bin/python
WANDB_API_KEY="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"

export WANDB_API_KEY
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

cd $PROJECT_ROOT

LOG_FILE=logs/olmo2_7B_keep12fresh2_resume200k_21.log

echo "=== keep12+fresh2 resume on .21 started at $(date) ===" | tee $LOG_FILE

DATA_PATH=/dev/shm/dolmino_now15b_zwfy6.npy
EXPECTED_SIZE=126907244672
MAX_WAIT=14400
WAITED=0
while true; do
  if [ -f "$DATA_PATH" ]; then
    CURRENT_SIZE=$(stat -c %s "$DATA_PATH" 2>/dev/null || echo 0)
    if [ "$CURRENT_SIZE" = "$EXPECTED_SIZE" ]; then break; fi
    echo "$(date): dolmino partial $CURRENT_SIZE/$EXPECTED_SIZE, waiting..." | tee -a $LOG_FILE
  fi
  sleep 120; WAITED=$((WAITED+120))
  if [ $WAITED -ge $MAX_WAIT ]; then echo "TIMEOUT waiting for dolmino"; exit 1; fi
done
DATA_SIZE=$(/opt/conda/envs/torch-base/bin/python -c "import numpy as np; d=np.load('$DATA_PATH', mmap_mode='r'); print(d.shape[0])" 2>/dev/null)
if [ "$DATA_SIZE" != "15491607" ]; then
  echo "ERROR: dataset rows=$DATA_SIZE expected 15491607" | tee -a $LOG_FILE
  exit 1
fi
echo "Data verified: $DATA_PATH rows=$DATA_SIZE" | tee -a $LOG_FILE

CKPT=$PROJECT_ROOT/outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt
if [ ! -f "$CKPT" ]; then
  echo "ERROR: $CKPT not found. PART 1 transfer incomplete?" | tee -a $LOG_FILE
  exit 1
fi
echo "Ckpt verified: $CKPT" | tee -a $LOG_FILE

torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --resume_from $CKPT \
  --keep_front_layers 12 \
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
  --output_dir $PROJECT_ROOT/outputs/olmo2_probe2_7B_keep12fresh2 \
  --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  2>&1 | tee -a $LOG_FILE
