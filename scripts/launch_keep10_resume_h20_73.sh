#!/bin/bash
# Launch keep10+fresh2 resume on .73 (8xH20, zwfy6 disk)
# Uses zwfy6 15.5M corpus (already in /dev/shm/dolmino_now15b.npy on .82)
# Resumes from step86500.pt (zwfy6 newest; wzc1 copy is stale at 83500 -- do NOT use it)
# eff_batch = batch_size(4) * grad_accum(4) * world_size(8) = 128

set -e

PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PYTHON=/opt/conda/envs/torch-base/bin/python
WANDB_API_KEY="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"

export WANDB_API_KEY
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

cd $PROJECT_ROOT

LOG_FILE=logs/olmo2_7B_keep10fresh2_resume200k_73.log

echo "=== keep10+fresh2 resume on .73 started at $(date) ===" | tee $LOG_FILE

DATA_PATH=/dev/shm/dolmino_now15b.npy
if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: $DATA_PATH not found" | tee -a $LOG_FILE
  exit 1
fi
DATA_SIZE=$($PYTHON -c "import numpy as np; d=np.load('$DATA_PATH', mmap_mode='r'); print(d.shape[0])")
if [ "$DATA_SIZE" != "15491607" ]; then
  echo "ERROR: dataset rows=$DATA_SIZE expected 15491607" | tee -a $LOG_FILE
  exit 1
fi
echo "Data verified: $DATA_PATH rows=$DATA_SIZE" | tee -a $LOG_FILE

CKPT=$PROJECT_ROOT/outputs/olmo2_probe2_7B_keep10fresh2/step86500.pt
if [ ! -f "$CKPT" ]; then
  echo "ERROR: $CKPT not found" | tee -a $LOG_FILE
  exit 1
fi
echo "Ckpt verified: $CKPT" | tee -a $LOG_FILE

torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --resume_from $CKPT \
  --keep_front_layers 10 \
  --n_fresh_layers 2 \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
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
  --keep_steps 83500,86500,100000,125000,150000,175000,200000 \
  --data_path $DATA_PATH \
  --output_dir $PROJECT_ROOT/outputs/olmo2_probe2_7B_keep10fresh2 \
  --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  2>&1 | tee -a $LOG_FILE
