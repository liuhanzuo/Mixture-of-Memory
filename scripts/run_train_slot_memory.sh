#!/bin/bash
# Slot Memory Training Launcher — 8 GPUs
set -e

PROJECT_DIR="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd $PROJECT_DIR

source .venv/bin/activate

STAGE=${1:-1}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/slot_memory_8gpu_stage${STAGE}_${TIMESTAMP}"

echo "[SlotMemory] Starting Stage ${STAGE} training at $(date)"
echo "[SlotMemory] Output: $OUTPUT_DIR"

# Resume from previous stage if available
RESUME_ARG=""
if [ "$STAGE" -gt 1 ]; then
    # Look for latest previous stage output
    PREV_STAGE=$((STAGE - 1))
    PREV_DIR=$(ls -d outputs/slot_memory_8gpu_stage${PREV_STAGE}_* 2>/dev/null | sort | tail -1)
    if [ -n "$PREV_DIR" ] && [ -d "$PREV_DIR/final" ]; then
        RESUME_ARG="--resume_from $PREV_DIR/final"
        echo "[SlotMemory] Resuming from $PREV_DIR/final"
    fi
fi

torchrun \
    --nproc_per_node=8 \
    --master_port=29506 \
    scripts/train_slot_memory.py \
    --stage $STAGE \
    --data data/rmt_train_mixed.jsonl \
    --output_dir $OUTPUT_DIR \
    --base_model ../models/Qwen--Qwen3-8b \
    --slot_dim 256 \
    --segment_length 1024 \
    --max_segments 4 \
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
    $RESUME_ARG

echo "[SlotMemory] Stage ${STAGE} training finished at $(date)"
