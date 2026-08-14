#!/bin/bash
# Launch keep8+fresh2 resume on .82 (8xH20, zwfy6 disk)
# DERIVED BY MECHANICAL SUBSTITUTION from launch_keep8_resume_h20_73.sh --
# the H20-PROVEN recipe -- so no hyperparameter can drift. Only these differ:
#   keep_front_layers, output_dir, log name, preflight arm guard.
# NOT derived from launch_keep12_resume_b200_21.sh: that is the B200 recipe
# (batch 8 x accum 2, wzc1 paths). keep12 is DEEPER than keep8, so batch 8 on a
# 97.8 GB H20 risks OOM. batch 4 x accum 4 keeps eff_batch = 4*4*8 = 128 identical.
# Uses zwfy6 15.5M corpus (/dev/shm/dolmino_now15b.npy on .82; 126.9 GB verified)
# Resumes from the DISCOVERED newest keep8 ckpt (see preflight); asserts >= step131000.
# eff_batch = batch_size(4) * grad_accum(4) * world_size(8) = 128

set -e

PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PYTHON=/opt/conda/envs/torch-base/bin/python
WANDB_API_KEY="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"

export WANDB_API_KEY
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

cd $PROJECT_ROOT

LOG_FILE=logs/olmo2_7B_keep8fresh2_resume200k_82_0814.log

echo "=== keep8+fresh2 resume on .82 started at $(date) ===" | tee $LOG_FILE

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

# ---- PREFLIGHT (2026-08-14): DISCOVER newest ckpt; refuse a stale/regressed one
# The pre-existing launch scripts pinned an OLD checkpoint filename. By 2026-08-14
# keep8 had reached step131000 and keep12 step166000, so running them as-written
# would have SILENTLY discarded ~10k and ~42k steps while producing a
# plausible-looking loss curve -- the same class of trap as the SparseForge
# dangling `last` symlink closed earlier today. Hence: discover, then assert.
NEWEST=$(ls -t $PROJECT_ROOT/outputs/olmo2_probe2_7B_keep8fresh2/step*.pt 2>/dev/null | head -1)
if [ -z "$NEWEST" ]; then
  echo "FATAL: no step*.pt in $PROJECT_ROOT/outputs/olmo2_probe2_7B_keep8fresh2" | tee -a $LOG_FILE; exit 1
fi
NEWSTEP=$(basename "$NEWEST" | tr -dc '0-9')
if [ "$NEWSTEP" -lt "131000" ]; then
  echo "FATAL: newest ckpt step$NEWSTEP < recorded 131000 -- refusing to lose steps" | tee -a $LOG_FILE; exit 1
fi
if [ "$NEWSTEP" -ge 200000 ]; then
  echo "DONE: step$NEWSTEP already >= max_steps 200000; nothing to resume" | tee -a $LOG_FILE; exit 0
fi
CKPT="$NEWEST"
echo "Resuming from DISCOVERED newest ckpt: $CKPT (step $NEWSTEP)" | tee -a $LOG_FILE

torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --resume_from $CKPT \
  --keep_front_layers 8 \
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
  --keep_steps 83500,121000,124000,150000,175000,200000 \
  --data_path $DATA_PATH \
  --output_dir $PROJECT_ROOT/outputs/olmo2_probe2_7B_keep8fresh2 \
  --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  2>&1 | tee -a $LOG_FILE
