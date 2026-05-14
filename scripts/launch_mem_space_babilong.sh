#!/usr/bin/env bash
# Memory-Space BABILong task-specific SFT (Fix 2).
#
# Default: DRY RUN — only prints the torchrun command.
# To actually launch, set DRY_RUN=0:
#
#   DRY_RUN=0 EXP_TAG=phase1 bash scripts/launch_mem_space_babilong.sh
#
# Continues training from the champion mem_space adapter on BABILong qa-tasks
# with a small PG-19 LM mix to mitigate catastrophic forgetting.

set -e

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory

export http_proxy=${http_proxy:-http://star-proxy.oa.com:3128}
export https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}
export PYTHONPATH=$(pwd)/third_party/babilong-pkg:$(pwd):${PYTHONPATH}
export PYTHONUNBUFFERED=1

# ---- knobs (override via env) ---------------------------------------------- #
DRY_RUN="${DRY_RUN:-1}"
EXP_TAG="${EXP_TAG:-babilong_phase1}"

BACKBONE_PATH="${BACKBONE_PATH:-models/Meta-Llama-3-8B-Instruct}"
INIT_CKPT="${INIT_CKPT:-outputs/champion_ckpt/mem_space_adapter.pt}"
INIT_CFG="${INIT_CFG:-outputs/champion_ckpt/adapter_config.json}"

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
LR="${LR:-1e-4}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

BABI_TASKS="${BABI_TASKS:-qa1}"
BABI_LENGTHS="${BABI_LENGTHS:-1k,2k}"
PG19_MIX="${PG19_MIX:-0.2}"
PG19_DATA="${PG19_DATA:-data/pg19_chunks_llama3.npy}"

# Set USE_CHAT_TEMPLATE=0 if backbone is base (non-Instruct).
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/babilong_sft_${EXP_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

# ---- assemble command ------------------------------------------------------ #
CHAT_FLAG=""
if [ "${USE_CHAT_TEMPLATE}" = "1" ]; then
  CHAT_FLAG="--use_chat_template"
fi

CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/train_mem_space_babilong.py \
  --model_path ${BACKBONE_PATH} \
  --init_checkpoint ${INIT_CKPT} \
  --init_adapter_config ${INIT_CFG} \
  --babilong_tasks ${BABI_TASKS} \
  --babilong_lengths ${BABI_LENGTHS} \
  --pg19_mix_fraction ${PG19_MIX} \
  --pg19_data ${PG19_DATA} \
  ${CHAT_FLAG} \
  --total_steps ${TOTAL_STEPS} \
  --lr ${LR} \
  --chunk_size ${CHUNK_SIZE} \
  --max_seq_len ${MAX_SEQ_LEN} \
  --output_dir ${OUTPUT_DIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[DRY RUN] command would be:"
  echo
  echo "${CMD}"
  echo
  echo "[DRY RUN] To launch, set DRY_RUN=0 (e.g. 'DRY_RUN=0 EXP_TAG=phase1 bash $0')."
  exit 0
fi

LOG_FILE="${LOG_DIR}/${EXP_TAG}_$(date +%Y%m%d_%H%M).log"
echo "[launch] logging to ${LOG_FILE}"
echo "[launch] ${CMD}"
exec ${CMD} 2>&1 | tee "${LOG_FILE}"
