#!/bin/bash
# Quick smoke test for RMT v8 direct injection — 1 GPU, 1 epoch, small sample
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export PYTHONUNBUFFERED=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/rmt_v8_debug_${TIMESTAMP}"

echo "[DEBUG v8] Starting at $(date)"
echo "[DEBUG v8] Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0 python scripts/train_rmt_v8.py \
  --data data/rmt_train_mixed.jsonl \
  --output_dir ${OUTPUT_DIR} \
  --base_model ../models/Qwen--Qwen3-8b \
  --num_memory_tokens 16 \
  --K_min 4 \
  --segment_length 1024 \
  --max_segments 3 \
  --num_epochs 1 \
  --lr 2e-5 \
  --rmt_lr 5e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --warmup_steps 10 \
  --log_every 5 \
  --save_every 9999 \
  --seed 42

echo "[DEBUG v8] Exit code: $?"
echo "[DEBUG v8] Finished at $(date)"
