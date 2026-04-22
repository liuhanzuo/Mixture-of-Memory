#!/bin/bash
# RMT v3 Scale-Up: 8-GPU training with LoRA rank 128, 10 epochs
# Targets 300-500 docs, minimum 25K steps, gate monitoring enabled
set -euo pipefail

cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

# ── GPU Verification ──────────────────────────────────────────────
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[Launcher] CUDA GPUs visible: $NUM_GPUS"
if [ "$NUM_GPUS" -lt 8 ]; then
    echo "[Launcher] WARNING: Expected 8 GPUs, found $NUM_GPUS"
    echo "[Launcher] Continuing with $NUM_GPUS GPUs..."
fi

# ── Configuration ─────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/rmt_v3_lora128_10ep_${TIMESTAMP}"
DATA_PATH="${DATA_PATH:-data/rmt_train_mixed.jsonl}"
MODEL_PATH="${MODEL_PATH:-../models/Qwen--Qwen3-8b}"

# Resume from checkpoint (uncomment to use):
# RESUME_FLAG="--resume_from outputs/rmt_v3_xxx/checkpoints/epoch_3"
RESUME_FLAG="${RESUME_FLAG:-}"

echo "[Launcher] Output: $OUTPUT_DIR"
echo "[Launcher] Data: $DATA_PATH"
echo "[Launcher] Model: $MODEL_PATH"
echo "[Launcher] LoRA rank: 128 (alpha=256)"
echo "[Launcher] Target: 10 epochs, min 25K steps"

# ── Launch ────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    scripts/train_rmt_v3.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_memory_tokens 8 \
    --segment_length 1024 \
    --max_segments 6 \
    --num_epochs 10 \
    --lr 1e-5 \
    --rmt_lr 5e-4 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --warmup_steps 2000 \
    --log_every 5 \
    --save_every 200 \
    --bottleneck_dim 32 \
    --seed 42 \
    $RESUME_FLAG

echo "[Launcher] Done. Output in $OUTPUT_DIR"
