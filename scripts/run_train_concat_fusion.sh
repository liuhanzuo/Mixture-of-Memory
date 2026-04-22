#!/bin/bash
# Launch script for ConcatFusionAttention (Phase 2) — 8×GPU via torchrun DDP.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
source "${PROJECT_DIR}/.venv/bin/activate"

# Config
BASE_MODEL="${BASE_MODEL:-${PROJECT_DIR}/models/Llama--Llama2-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sparse_memory_concat_fusion_v1}"
DATA_PATH="${DATA_PATH:-/tmp/pg19_small.jsonl}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-5000}"
BYPASS_BIAS_INIT="${BYPASS_BIAS_INIT:--2.0}"
BYPASS_GATE_LR="${BYPASS_GATE_LR:-10.0}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SEQ_LEN="${SEQ_LEN:-1024}"

echo "=== Phase 2: Concat Fusion + Bypass Gate ==="
echo "Base model:   ${BASE_MODEL}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Data path:    ${DATA_PATH}"
echo "GPUs:         ${GPUS}"
echo "Bypass init:  ${BYPASS_BIAS_INIT}"
echo "Bypass LR:    ${BYPASS_GATE_LR}x base LR"
echo "Max steps:    ${MAX_STEPS}"

# Kill any lingering torchrun on same port
fuser -k ${MASTER_PORT}/tcp 2>/dev/null || true
sleep 2

torchrun \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_gated_sparse_memory.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${BASE_MODEL}" \
  --data_path "${DATA_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --num_slots 128 \
  --top_k 8 \
  --window_size 256 \
  --bypass_bias_init "${BYPASS_BIAS_INIT}" \
  --bypass_gate_lr_multiplier "${BYPASS_GATE_LR}" \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 50 \
  --save_steps 1000 \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --max_seq_length "${SEQ_LEN}" \
  --overwrite_output_dir \
  --ddp_find_unused_parameters 

echo "=== Training complete ==="
