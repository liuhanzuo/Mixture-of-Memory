#!/usr/bin/env bash
# Phase-1B v6 — replace writeback ablation on raw Llama-3.2-1B-Instruct.
#
# Experiment: slot ← s_new (no EMA) for ALL selected slots.
# All other hyperparameters identical to the v2 baseline recipe.
#
# Purpose: test whether removing EMA and writing s_new directly improves
# counting/tracking tasks (qa7-qa10 on BABILong) at the cost of potentially
# degrading long-range retrieval (qa1, qa5).
#
# Launch:  bash scripts/launch_phase1b_v6_replace_writeback.sh
# Outputs: outputs/babilong_sft_phase1b_v6_replace_writeback_<timestamp>/
# Log:     logs/phase1b_v6_replace_writeback_<timestamp>.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

EXP_TAG="${EXP_TAG:-phase1b_v6_replace_writeback}"
BACKBONE_PATH="${BACKBONE_PATH:-models/Llama-3.2-1B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
LR="${LR:-2e-5}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

BABI_TASKS="${BABI_TASKS:-qa1,qa2,qa5}"
BABI_LENGTHS="${BABI_LENGTHS:-1k,2k,4k}"
PG19_MIX="${PG19_MIX:-0.2}"
PG19_DATA="${PG19_DATA:-data/pg19_chunks_llama3.npy}"

# v2-identical architecture params
NUM_SLOTS="${NUM_SLOTS:-512}"
TOP_K="${TOP_K:-64}"
SELECTOR_DIM="${SELECTOR_DIM:-128}"
SELECTOR_TEMP="${SELECTOR_TEMP:-1.0}"
WB_GATE_MAX="${WB_GATE_MAX:-0.3}"
WB_WARMUP_STEPS="${WB_WARMUP_STEPS:-1000}"
LOAD_BAL_W="${LOAD_BAL_W:-0.01}"
ENTROPY_W="${ENTROPY_W:-0.001}"
KEY_REP_W="${KEY_REP_W:-0.05}"
KEY_REP_TH="${KEY_REP_TH:-0.3}"
PEAK_ROUTE_W="${PEAK_ROUTE_W:-0.05}"
SLOT_NORM_CAP="${SLOT_NORM_CAP:-5.0}"
SLOT_INIT="${SLOT_INIT:-random}"
SLOT_INIT_NOISE="${SLOT_INIT_NOISE:-0.05}"
FORGET_BIAS="${FORGET_BIAS:-2.0}"
INPUT_BIAS="${INPUT_BIAS:-0.0}"
L3_N_SUMMARY="${L3_N_SUMMARY:-64}"
L3_N_LAYERS="${L3_N_LAYERS:-2}"
L3_N_HEADS="${L3_N_HEADS:-8}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/babilong_sft_${EXP_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

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
  --max_seq_len ${MAX_SEQ_LEN} \
  --save_interval ${SAVE_INTERVAL} \
  --num_slots ${NUM_SLOTS} \
  --top_k ${TOP_K} \
  --selector_dim ${SELECTOR_DIM} \
  --selector_temperature ${SELECTOR_TEMP} \
  --writeback_gate_max ${WB_GATE_MAX} \
  --writeback_warmup_steps ${WB_WARMUP_STEPS} \
  --load_balance_weight ${LOAD_BAL_W} \
  --entropy_aux_weight ${ENTROPY_W} \
  --key_repulsion_weight ${KEY_REP_W} \
  --key_repulsion_threshold ${KEY_REP_TH} \
  --peak_routing_weight ${PEAK_ROUTE_W} \
  --slot_value_norm_cap ${SLOT_NORM_CAP} \
  --slot_init ${SLOT_INIT} \
  --slot_init_noise ${SLOT_INIT_NOISE} \
  --use_dual_gate \
  --forget_bias_init ${FORGET_BIAS} \
  --input_bias_init ${INPUT_BIAS} \
  --use_l3_summary \
  --l3_n_summary ${L3_N_SUMMARY} \
  --l3_n_layers ${L3_N_LAYERS} \
  --l3_n_heads ${L3_N_HEADS} \
  --shared_memory_bank \
  --use_replace_writeback \
  --output_dir ${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/${EXP_TAG}_$(date +%Y%m%d_%H%M).log"
echo "[launch] logging to ${LOG_FILE}"
echo "[launch] ${CMD}"
exec ${CMD} 2>&1 | tee "${LOG_FILE}"
