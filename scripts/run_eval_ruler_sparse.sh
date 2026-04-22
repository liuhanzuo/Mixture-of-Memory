#!/bin/bash
# Run RULER CWE evaluation for the sparse memory trained model at 64K context.
#
# Usage:
#   bash scripts/run_eval_ruler_sparse.sh [GPU_ID] [CHECKPOINT]
#
# Default: GPU 1, checkpoint at outputs/sparse_memory_ablation_8gpu_v5/final
#
# Pass CHECKPOINT as a relative path from project root, e.g.:
#   bash scripts/run_eval_ruler_sparse.sh 1 outputs/sparse_memory_ablation_8gpu_v5/checkpoint-5000

set -euo pipefail

GPU_ID="${1:-1}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${2:-outputs/sparse_memory_ablation_8gpu_v5/final}"
MODEL_PATH="${PROJECT_ROOT}/${CHECKPOINT}"
OUTPUT_DIR="${PROJECT_ROOT}/results"
MODEL_NAME="$(basename "${CHECKPOINT}")"
OUTPUT_FILE="${OUTPUT_DIR}/ruler_cwe_64k_${MODEL_NAME}.json"

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
echo "RULER CWE Evaluation — Sparse Memory Model"
echo "=========================================="
echo "Model:          ${MODEL_PATH}"
echo "Context length: ${CONTEXT_LENGTH}"
echo "GPU:            ${GPU_ID}"
echo "Samples:        ${NUM_SAMPLES}"
echo "Output:         ${OUTPUT_FILE}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python "${PROJECT_ROOT}/scripts/eval_ruler_cwe.py" \
    --model_path "${MODEL_PATH}" \
    --max_seq_length ${CONTEXT_LENGTH} \
    --num_samples ${NUM_SAMPLES} \
    --num_cw ${NUM_CW} \
    --freq_cw ${FREQ_CW} \
    --freq_ucw ${FREQ_UCW} \
    --seed ${SEED} \
    --tokens_to_generate ${MAX_NEW_TOKENS} \
    --apply_chat_template \
    --output_file "${OUTPUT_FILE}"

echo "Done. Results at ${OUTPUT_FILE}"
