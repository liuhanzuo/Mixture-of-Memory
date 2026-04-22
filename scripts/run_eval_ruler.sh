#!/bin/bash
# Run RULER CWE baseline evaluation for Qwen3-8B base model at 64K context.
#
# Usage:
#   bash scripts/run_eval_ruler.sh [GPU_ID]
#
# Default: GPU 0, 50 samples, 64K context length.

set -euo pipefail

GPU_ID="${1:-1}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen--Qwen3-8b"
OUTPUT_DIR="${PROJECT_ROOT}/results"
OUTPUT_FILE="${OUTPUT_DIR}/ruler_cwe_64k_qwen3-8b_base.json"

# Config
CONTEXT_LENGTH=65536
NUM_SAMPLES=50
NUM_CW=10
FREQ_CW=30
FREQ_UCW=3
SEED=42
MAX_NEW_TOKENS=150

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "RULER CWE Baseline Evaluation"
echo "=========================================="
echo "Model:          ${MODEL_PATH}"
echo "Context length: ${CONTEXT_LENGTH}"
echo "GPU:            ${GPU_ID}"
echo "Samples:        ${NUM_SAMPLES}"
echo "Output:         ${OUTPUT_FILE}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python "${PROJECT_ROOT}/scripts/eval_ruler_baseline.py" \
    --model_path "${MODEL_PATH}" \
    --context_length ${CONTEXT_LENGTH} \
    --num_samples ${NUM_SAMPLES} \
    --num_cw ${NUM_CW} \
    --freq_cw ${FREQ_CW} \
    --freq_ucw ${FREQ_UCW} \
    --seed ${SEED} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --gpu_id 0 \
    --output "${OUTPUT_FILE}"

echo "Done. Results at ${OUTPUT_FILE}"
