#!/bin/bash
# RMT Training Launcher - 8 GPUs with gradient checkpointing
set -e

PROJECT_DIR="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd $PROJECT_DIR

source .venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/rmt_v1_${TIMESTAMP}"

echo "[RMT] Starting training at $(date)"
echo "[RMT] Output: $OUTPUT_DIR"

torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    scripts/train_rmt.py \
    --model_path ../models/Qwen--Qwen3-8b \
    --data_path data/rmt_train_wikitext.jsonl \
    --output_dir $OUTPUT_DIR \
    --num_memory_tokens 64 \
    --segment_length 2048 \
    --max_total_tokens 12288 \
    --num_memory_heads 8 \
    --num_epochs 3 \
    --lr 1e-5 \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --warmup_steps 100 \
    --gradient_checkpointing \
    --log_every 20 \
    --save_every 200 \
    --seed 42

echo "[RMT] Training finished at $(date)"
