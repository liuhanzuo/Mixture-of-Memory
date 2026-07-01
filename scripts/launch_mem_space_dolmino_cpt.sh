#!/usr/bin/env bash
# Dolmino Continued Pretraining (CPT) for Memory-Space adapter.
#
# Purpose: Train the mem_space memory adapter on diverse text (Dolmino, 0.47B tokens)
# with curriculum learning, following the MemoryLLM methodology. The adapter learns
# to compress diverse text into memory, generalizing beyond BABILong QA tasks.
#
# Curriculum schedule:
#   Step 0:      n_ctx=1   (2K effective context)
#   Step 10000:  n_ctx=2   (3K effective context)
#   Step 15000:  n_ctx=4   (5K effective context)
#   Step 25000:  n_ctx=8   (9K effective context)
#   Step 40000:  n_ctx=16  (17K effective context)
#
# Launch (local H20 8-GPU):
#   bash scripts/launch_mem_space_dolmino_cpt.sh
#
# Target node: Local 8x H20 (97.8 GiB)
# Outputs: outputs/dolmino_cpt_v1/
# Log:     logs/dolmino_cpt_v1_<timestamp>.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

# Proxy for HuggingFace access (BABILong dataset download)
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Python binary
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# ────────────────────────────────────────────────────────────────────────────
# Experiment config
# ────────────────────────────────────────────────────────────────────────────

EXP_TAG="${EXP_TAG:-dolmino_cpt_v1}"

# Model: use BASE (not Instruct) for CPT
BACKBONE="${BACKBONE:-models/Meta-Llama-3-8B}"

# Data
DOLMINO_PATH="${DOLMINO_PATH:-MemLong/data/processed/dolmino_0.5B_1024/train}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"

# Training
NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-60000}"
LR="${LR:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"

# Curriculum: "step:n_ctx,step:n_ctx,..."
CURRICULUM="${CURRICULUM:-0:1,10000:2,15000:4,25000:8,40000:16}"

# BABILong mix (15% of steps are BABILong SFT to maintain retrieval ability)
BABI_MIX="${BABI_MIX:-0.15}"
BABI_TASKS="${BABI_TASKS:-qa1,qa2,qa5}"
BABI_LENGTHS="${BABI_LENGTHS:-0k,1k,2k,4k}"

# ────────────────────────────────────────────────────────────────────────────
# Memory-Space architecture config (matches P8 champion recipe)
# ────────────────────────────────────────────────────────────────────────────

NUM_SLOTS="${NUM_SLOTS:-512}"
TOP_K="${TOP_K:-64}"
SELECTOR_DIM="${SELECTOR_DIM:-128}"
SELECTOR_TEMP="${SELECTOR_TEMP:-20.0}"
WB_GATE_MAX="${WB_GATE_MAX:-0.3}"
WB_WARMUP_STEPS="${WB_WARMUP_STEPS:-0}"
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

# Optional: warm-start from existing adapter
INIT_CKPT="${INIT_CKPT:-}"
INIT_CONFIG="${INIT_CONFIG:-}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ────────────────────────────────────────────────────────────────────────────
# Build command
# ────────────────────────────────────────────────────────────────────────────

CMD="$PYTHON_BIN -m torch.distributed.run --nproc_per_node=${NUM_GPUS} \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path ${BACKBONE} \
  --dolmino_path ${DOLMINO_PATH} \
  --chunk_size ${CHUNK_SIZE} \
  --curriculum '${CURRICULUM}' \
  --babilong_mix_fraction ${BABI_MIX} \
  --babilong_tasks ${BABI_TASKS} \
  --babilong_lengths ${BABI_LENGTHS} \
  --total_steps ${TOTAL_STEPS} \
  --lr ${LR} \
  --warmup_steps ${WARMUP_STEPS} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --save_interval ${SAVE_INTERVAL} \
  --eval_interval ${EVAL_INTERVAL} \
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
  --gradient_checkpointing \
  --output_dir ${OUTPUT_DIR}"

# Add optional warm-start flags
if [ -n "${INIT_CKPT}" ]; then
  CMD="${CMD} --init_checkpoint ${INIT_CKPT}"
fi
if [ -n "${INIT_CONFIG}" ]; then
  CMD="${CMD} --init_adapter_config ${INIT_CONFIG}"
fi

# ────────────────────────────────────────────────────────────────────────────
# Launch
# ────────────────────────────────────────────────────────────────────────────

LOG_FILE="${LOG_DIR}/${EXP_TAG}_$(date +%Y%m%d_%H%M).log"
echo "[launch] Dolmino CPT — ${EXP_TAG}"
echo "[launch] backbone=${BACKBONE} (base model)"
echo "[launch] dolmino=${DOLMINO_PATH} curriculum=${CURRICULUM}"
echo "[launch] babilong_mix=${BABI_MIX} tasks=${BABI_TASKS} lengths=${BABI_LENGTHS}"
echo "[launch] GPUs=${NUM_GPUS} total_steps=${TOTAL_STEPS} lr=${LR} grad_accum=${GRAD_ACCUM}"
echo "[launch] mem_space: slots=${NUM_SLOTS} top_k=${TOP_K} selector_dim=${SELECTOR_DIM} temp=${SELECTOR_TEMP}"
echo "[launch] dual_gate: forget_bias=${FORGET_BIAS} input_bias=${INPUT_BIAS}"
echo "[launch] L3: n_summary=${L3_N_SUMMARY} n_layers=${L3_N_LAYERS} n_heads=${L3_N_HEADS}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] log=${LOG_FILE}"
echo "[launch] ${CMD}"
echo ""

exec ${CMD} 2>&1 | tee "${LOG_FILE}"
