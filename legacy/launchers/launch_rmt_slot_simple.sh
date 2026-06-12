#!/bin/bash
# Simple launch script for RMT-Slot on b200-2

set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

# Configuration
MODEL_PATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
DATA_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
OUTPUT_DIR="outputs/rmt_slot_simple"

# Create output dir
mkdir -p "$OUTPUT_DIR"

# Launch with minimal arguments
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train_rmt_slot.py \
    --model "$MODEL_PATH" \
    --shard_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_slots 64 \
    --top_k 8 \
    --segment_length 1024 \
    --max_n_segments 4 \
    --max_steps 2000 \
    --lr 5e-6 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 200 \
    --eval_interval 200 \
    --save_interval 500 \
    --seed 42 \
    2>&1 | tee "logs/rmt_slot_simple_$(date +%Y%m%d_%H%M).log"

echo "RMT-Slot simple launched on b200-2"
