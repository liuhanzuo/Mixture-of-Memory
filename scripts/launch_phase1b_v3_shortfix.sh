#!/usr/bin/env bash
# Phase-1B v3 short-fix — fix the qa5 short-context regression observed in v2.
#
# v2 baseline (outputs/babilong_sft_phase1b_v2_10k/):
#   * 10k steps, lr=2e-5, lengths=1k,2k,4k uniform, pg19_mix=0.20
#   * Result: 21-cell mean=37.29; qa5 0K-4K avg ~41.5% (20.5pp BELOW LM2-paper
#     Llama-3.2-1.2B vanilla 62%); qa5 ≥8K wins by +19.2pp.
#
# Hypothesis: BABILong train mix is biased away from short-context samples,
# so mem_space "always uses memory" even when the question fits in window
# and standard self-attn would be sufficient.
#
# v3 changes vs v2 (NO ckpt/eval-CSV touched):
#   1.  Add length=0k (pure in-context, no distractor) to the train mix.
#   2.  Oversample short cells: weights 0k=3, 1k=3, 2k=1, 4k=1
#       (37.5% 0k + 37.5% 1k + 12.5% 2k + 12.5% 4k = 75% short).
#   3.  Skip L1/L3 writeback when total_len <= chunk_size (1024) so the
#       model learns to bypass memory at short range.
#   4.  Total 5000 steps (v2 showed diminishing returns after ~5k).
#   5.  Save every 500 steps.
#
# Launch: bash scripts/launch_phase1b_v3_shortfix.sh
# Outputs: outputs/babilong_sft_phase1b_v3_shortfix/
# Log:     logs/phase1b_v3_shortfix_<timestamp>.log

set -e

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory

export http_proxy=${http_proxy:-http://star-proxy.oa.com:3128}
export https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}
export PYTHONPATH=$(pwd)/third_party/babilong-pkg:$(pwd):${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

# ---- knobs (override via env) ---------------------------------------------- #
EXP_TAG="${EXP_TAG:-phase1b_v3_shortfix}"

BACKBONE_PATH="${BACKBONE_PATH:-models/Llama-3.2-1B-Instruct}"

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
LR="${LR:-2e-5}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

BABI_TASKS="${BABI_TASKS:-qa1,qa2,qa5}"
BABI_LENGTHS="${BABI_LENGTHS:-0k,1k,2k,4k}"
BABI_LEN_WEIGHTS="${BABI_LEN_WEIGHTS:-3,3,1,1}"
PG19_MIX="${PG19_MIX:-0.2}"
PG19_DATA="${PG19_DATA:-data/pg19_chunks_llama3.npy}"

# v2 architecture hyperparams (do NOT change — replicating v2 exactly except
# for the data sampler + skip-mem-when-short hooks).
NUM_SLOTS="${NUM_SLOTS:-512}"
TOP_K="${TOP_K:-64}"
SELECTOR_DIM="${SELECTOR_DIM:-128}"
WB_GATE_MAX="${WB_GATE_MAX:-0.3}"
WB_WARMUP_STEPS="${WB_WARMUP_STEPS:-1000}"
LOAD_BAL_W="${LOAD_BAL_W:-0.01}"
ENTROPY_W="${ENTROPY_W:-0.001}"
KEY_REP_W="${KEY_REP_W:-0.05}"
KEY_REP_TH="${KEY_REP_TH:-0.3}"
PEAK_ROUTE_W="${PEAK_ROUTE_W:-0.05}"
SLOT_NORM_CAP="${SLOT_NORM_CAP:-5.0}"
FORGET_BIAS="${FORGET_BIAS:-2.0}"
INPUT_BIAS="${INPUT_BIAS:-0.0}"
L3_N_SUMMARY="${L3_N_SUMMARY:-64}"
L3_N_LAYERS="${L3_N_LAYERS:-2}"
L3_N_HEADS="${L3_N_HEADS:-8}"

USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/babilong_sft_${EXP_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

CHAT_FLAG=""
if [ "${USE_CHAT_TEMPLATE}" = "1" ]; then
  CHAT_FLAG="--use_chat_template"
fi

CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/train_mem_space_babilong.py \
  --model_path ${BACKBONE_PATH} \
  --babilong_tasks ${BABI_TASKS} \
  --babilong_lengths ${BABI_LENGTHS} \
  --babilong_length_weights ${BABI_LEN_WEIGHTS} \
  --skip_mem_when_short \
  --pg19_mix_fraction ${PG19_MIX} \
  --pg19_data ${PG19_DATA} \
  ${CHAT_FLAG} \
  --total_steps ${TOTAL_STEPS} \
  --lr ${LR} \
  --chunk_size ${CHUNK_SIZE} \
  --max_seq_len ${MAX_SEQ_LEN} \
  --save_interval ${SAVE_INTERVAL} \
  --num_slots ${NUM_SLOTS} \
  --top_k ${TOP_K} \
  --selector_dim ${SELECTOR_DIM} \
  --writeback_gate_max ${WB_GATE_MAX} \
  --writeback_warmup_steps ${WB_WARMUP_STEPS} \
  --load_balance_weight ${LOAD_BAL_W} \
  --entropy_aux_weight ${ENTROPY_W} \
  --key_repulsion_weight ${KEY_REP_W} \
  --key_repulsion_threshold ${KEY_REP_TH} \
  --peak_routing_weight ${PEAK_ROUTE_W} \
  --slot_value_norm_cap ${SLOT_NORM_CAP} \
  --use_dual_gate \
  --forget_bias_init ${FORGET_BIAS} \
  --input_bias_init ${INPUT_BIAS} \
  --use_l3_summary \
  --l3_n_summary ${L3_N_SUMMARY} \
  --l3_n_layers ${L3_N_LAYERS} \
  --l3_n_heads ${L3_N_HEADS} \
  --output_dir ${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/${EXP_TAG}_$(date +%Y%m%d_%H%M).log"
echo "[launch] logging to ${LOG_FILE}"
echo "[launch] ${CMD}"
exec ${CMD} 2>&1 | tee "${LOG_FILE}"
