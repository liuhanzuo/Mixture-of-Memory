#!/bin/bash
# Launch script for SparseMemory training on 8×GPU via torchrun DDP.
# Override defaults via environment variables.

set -euo pipefail

# ── Configurable paths ─────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"

source "${PROJECT_DIR}/.venv/bin/activate"

BASE_MODEL="${BASE_MODEL:-${PROJECT_DIR}/models/Llama--Llama2-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sparse_memory}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/pg19_train.jsonl}"

# ── Memory hyperparameters ─────────────────────────────────────────────────
MEMORY_SLOTS="${MEMORY_SLOTS:-128}"
TOP_K="${TOP_K:-8}"
SLIDING_WINDOW="${SLIDING_WINDOW:-256}"
EMA_ALPHA="${EMA_ALPHA:-0.1}"
GATE_BIAS_INIT="${GATE_BIAS_INIT:-2.0}"

# ── Mixed precision ────────────────────────────────────────────────────────
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # bf16 (default), fp16, or none

# ── Training hyperparameters ───────────────────────────────────────────────
LR="${LR:-2e-5}"
SEQ_LEN="${SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-5000}"
LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

# ── Launch ─────────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"

echo "=== SparseMemory Training Launch ==="
echo "Base model:  ${BASE_MODEL}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Data path:   ${DATA_PATH}"
echo "GPUs:        ${NUM_GPUS} × ${NNODES} node(s)"
echo "Memory:      slots=${MEMORY_SLOTS} top_k=${TOP_K} window=${SLIDING_WINDOW}"
echo "Precision:   ${MIXED_PRECISION}"
echo ""

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NUM_GPUS}" \
    "${PROJECT_DIR}/scripts/train_sparse_memory.py" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_path "${DATA_PATH}" \
    --memory_slots "${MEMORY_SLOTS}" \
    --top_k "${TOP_K}" \
    --sliding_window "${SLIDING_WINDOW}" \
    --ema_alpha "${EMA_ALPHA}" \
    --gate_bias_init "${GATE_BIAS_INIT}" \
    --lr "${LR}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accumulation_steps "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --log_every "${LOG_EVERY}" \
    --save_every "${SAVE_EVERY}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --gradient_checkpointing
