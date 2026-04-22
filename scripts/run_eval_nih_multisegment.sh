#!/bin/bash
# Run multi-segment NIH evaluation for slot memory validation
#
# This script tests whether slot memory preserves information across segment boundaries
# by comparing multi-segment accuracy against single-segment baseline.
#
# Usage:
#   bash scripts/run_eval_nih_multisegment.sh [gpu_id]
#
# Arguments:
#   gpu_id: GPU device ID (default: 1)

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b}"
OUTPUT_DIR="${OUTPUT_DIR:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/nih_multisegment}"

# Checkpoint paths (Stage 2 and Stage 3)
STAGE2_PATH="${STAGE2_PATH:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/slot_memory_8gpu_stage2_20260420_122501_stage2_20260420_122538/final}"
STAGE3_PATH="${STAGE3_PATH:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/slot_memory_8gpu_stage3_20260420_164731_stage3_20260420_164811/final}"

# Runtime parameters
DEVICE="${DEVICE:-cuda:${1:-1}}"
NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_SLOTS="${NUM_SLOTS:-16}"
SLOT_DIM="${SLOT_DIM:-256}"

# Change to MoM directory
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

echo "========================================"
echo "Multi-Segment NIH Evaluation"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Stage 2: ${STAGE2_PATH}"
echo "Stage 3: ${STAGE3_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "Trials: ${NUM_TRIALS}"
echo "Slots: ${NUM_SLOTS} × ${SLOT_DIM} dims"
echo "========================================"
echo ""

# Check if checkpoints exist
if [ ! -d "${STAGE2_PATH}" ] && [ ! -d "${STAGE3_PATH}" ]; then
    echo "ERROR: Neither Stage 2 nor Stage 3 checkpoint directory exists!"
    echo "Stage 2: ${STAGE2_PATH}"
    echo "Stage 3: ${STAGE3_PATH}"
    exit 1
fi

if [ ! -d "${STAGE2_PATH}" ]; then
    echo "WARNING: Stage 2 checkpoint not found, skipping..."
    STAGE2_PATH=""
fi

if [ ! -d "${STAGE3_PATH}" ]; then
    echo "WARNING: Stage 3 checkpoint not found, skipping..."
    STAGE3_PATH=""
fi

# Run evaluation
python scripts/eval_nih_multisegment.py \
    --model_path "${MODEL_PATH}" \
    --stage2_path "${STAGE2_PATH}" \
    --stage3_path "${STAGE3_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --num_trials "${NUM_TRIALS}" \
    --seed 42 \
    --num_slots "${NUM_SLOTS}" \
    --slot_dim "${SLOT_DIM}" \
    --skip_baseline

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
