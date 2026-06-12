#!/bin/bash
# Launch RMT-Slot hybrid on b200-2 (28.89.17.144)

set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

# Configuration
MODEL_PATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
DATA_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
OUTPUT_DIR="outputs/rmt_slot_medium"
LOG_FILE="logs/rmt_slot_medium_$(date +%Y%m%d_%H%M).log"

# Hyperparameters
NUM_SLOTS=64
TOP_K=8
SEGMENT_LENGTH=1024
MAX_STEPS=2000
LR=5e-6
GRAD_ACCUM=4
WARMUP=200

# Create output dir
mkdir -p "$OUTPUT_DIR"

# Launch with torchrun
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train_rmt_slot.py \
    --model "$MODEL_PATH" \
    --shard_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_slots "$NUM_SLOTS" \
    --top_k "$TOP_K" \
    --segment_length "$SEGMENT_LENGTH" \
    --max_steps "$MAX_STEPS" \
    --learning_rate "$LR" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --warmup_steps "$WARMUP" \
    --eval_interval 200 \
    --save_interval 500 \
    --logging_steps 10 \
    --with_tracking \
    --report_to tensorboard \
    --bf16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 4 \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo "RMT-Slot launched on b200-2. Log: $LOG_FILE"
