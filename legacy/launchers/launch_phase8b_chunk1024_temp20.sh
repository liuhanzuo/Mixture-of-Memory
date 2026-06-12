#!/usr/bin/env bash
# Phase-8B: chunk1024 + selector_temperature=20 — combining the two best levers.
#
# Rationale: chunk1024 alone scored 51.23, P8+temp20 scored 54.62.
# The main gap is selector_temperature (1.0 vs 20.0). Combining both
# should yield the best result by enabling sharper routing + larger chunk coverage.
#
# Target: Local H20 (8x H20, 97.8 GiB)
# Outputs: outputs/babilong_sft_phase8b_chunk1024_temp20_<timestamp>/
# Log:     logs/phase8b_chunk1024_temp20_<timestamp>.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
BACKBONE_PATH="${BACKBONE_PATH:-models/Meta-Llama-3-8B-Instruct}"
NUM_GPUS="${NUM_GPUS:-8}"

# Key combination: chunk_size=1024 (from scale_chunk1024) + temp=20 (from P8+temp20)
CHUNK_SIZE=1024
SELECTOR_TEMP=20.0
TOP_K=64
NUM_SLOTS=512
TOTAL_STEPS=5000
LR=2e-5

BABI_TASKS="qa1,qa2,qa5"
BABI_LENGTHS="1k,2k,4k"
PG19_MIX=0.2
PG19_DATA="data/pg19_chunks_llama3.npy"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_TAG="phase8b_chunk1024_temp20_${TIMESTAMP}"
OUTPUT_DIR="outputs/babilong_sft_${EXP_TAG}"
LOG_FILE="logs/${EXP_TAG}.log"
mkdir -p "$OUTPUT_DIR" logs

CMD="$PYTHON_BIN -m torch.distributed.run --nproc_per_node=${NUM_GPUS} scripts/train_mem_space_babilong.py \
  --model_path ${BACKBONE_PATH} \
  --babilong_tasks ${BABI_TASKS} \
  --babilong_lengths ${BABI_LENGTHS} \
  --pg19_mix_fraction ${PG19_MIX} \
  --pg19_data ${PG19_DATA} \
  --use_chat_template \
  --total_steps ${TOTAL_STEPS} \
  --lr ${LR} \
  --chunk_size ${CHUNK_SIZE} \
  --max_seq_len 4096 \
  --save_interval 500 \
  --num_slots ${NUM_SLOTS} \
  --top_k ${TOP_K} \
  --selector_dim 128 \
  --selector_temperature ${SELECTOR_TEMP} \
  --writeback_gate_max 0.3 \
  --writeback_warmup_steps 1000 \
  --load_balance_weight 0.01 \
  --entropy_aux_weight 0.001 \
  --key_repulsion_weight 0.05 \
  --key_repulsion_threshold 0.3 \
  --peak_routing_weight 0.05 \
  --slot_value_norm_cap 5.0 \
  --slot_init random \
  --slot_init_noise 0.05 \
  --use_dual_gate \
  --forget_bias_init 2.0 \
  --input_bias_init 0.0 \
  --use_l3_summary \
  --l3_n_summary 64 \
  --l3_n_layers 2 \
  --l3_n_heads 8 \
  --shared_memory_bank \
  --gradient_checkpointing \
  --output_dir ${OUTPUT_DIR}"

echo "[launch] chunk1024 + temp20 — logging to ${LOG_FILE}"
echo "[launch] selector_temperature=${SELECTOR_TEMP}, chunk_size=${CHUNK_SIZE}, top_k=${TOP_K}"
echo "[launch] ${CMD}"
exec ${CMD} 2>&1 | tee "${LOG_FILE}"
