#!/bin/bash
set -e

PROJECT_DIR="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd $PROJECT_DIR
source .venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/slot_memory_8gpu_stage2_${TIMESTAMP}"

echo "[SlotMemory] Starting Stage 2 at $(date)"
echo "[SlotMemory] Output: $OUTPUT_DIR"
echo "[SlotMemory] Resume from: outputs/slot_memory_8gpu_20260419_143943_20260419_144019/stage1/final"

torchrun \
    --nproc_per_node=8 \
    --master_port=29506 \
    scripts/train_slot_memory.py \
    --stage 2 \
    --data data/rmt_train_mixed.jsonl \
    --output_dir $OUTPUT_DIR \
    --base_model ../models/Qwen--Qwen3-8b \
    --slot_dim 256 \
    --segment_length 1024 \
    --max_segments 6 \
    --bptt_depth 2 \
    --learning_rate 2e-5 \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --warmup_steps 30 \
    --log_every 10 \
    --save_every 200 \
    --lora_r 16 \
    --lora_alpha 32 \
    --ddp \
    --seed 42 \
    --resume_from outputs/slot_memory_8gpu_20260419_143943_20260419_144019/stage1/final \
    2>&1 | tee ${OUTPUT_DIR}.log

echo "[SlotMemory] Stage 2 finished at $(date)"
