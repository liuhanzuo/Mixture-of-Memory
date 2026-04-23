#!/bin/bash
# DMS Retrofitting Training Launcher
# Launches training with 4x or 8x compression ratio on Llama-2-7B
#
# Usage:
#   bash scripts/launch_dms_train.sh 8    # 8x compression
#   bash scripts/launch_dms_train.sh 4    # 4x compression
#
set -euo pipefail

COMPRESSION_RATIO=${1:-8}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/dms_${COMPRESSION_RATIO}x"
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)

echo "=== DMS Retrofitting Training ==="
echo "Compression Ratio: ${COMPRESSION_RATIO}x"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"

# DMS paper: 100 steps per unit of CR
NUM_STEPS=$((COMPRESSION_RATIO * 100))

# Multi-GPU via torchrun
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Launching with ${NUM_GPUS} GPUs via torchrun..."
    torchrun --nproc_per_node="${NUM_GPUS}" \
        "${PROJECT_ROOT}/scripts/train_dms.py" \
        --model_name_or_path Qwen/Qwen3-8B \
        --compression_ratio "${COMPRESSION_RATIO}" \
        --num_train_steps "${NUM_STEPS}" \
        --output_dir "${OUTPUT_DIR}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --max_seq_length 2048 \
        --sliding_window 256 \
        --warmup_steps 100 \
        --bf16 \
        --logging_steps 10 \
        --save_steps 200 \
        --num_gpus "${NUM_GPUS}" \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
else
    echo "Launching single-GPU training..."
    python "${PROJECT_ROOT}/scripts/train_dms.py" \
        --model_name_or_path Qwen/Qwen3-8B \
        --compression_ratio "${COMPRESSION_RATIO}" \
        --num_train_steps "${NUM_STEPS}" \
        --output_dir "${OUTPUT_DIR}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --max_seq_length 2048 \
        --sliding_window 256 \
        --warmup_steps 100 \
        --bf16 \
        --logging_steps 10 \
        --save_steps 200 \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

echo "Training complete. Evaluating..."

# Evaluate
python "${PROJECT_ROOT}/scripts/eval_dms.py" \
    --model_path "${OUTPUT_DIR}/final" \
    --baseline_model_path Qwen/Qwen3-8B \
    --compression_ratio "${COMPRESSION_RATIO}" \
    --output_file "${OUTPUT_DIR}/eval_results.json"

echo "Done! Results in ${OUTPUT_DIR}/eval_results.json"
