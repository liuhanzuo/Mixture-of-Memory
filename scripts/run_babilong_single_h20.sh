#!/usr/bin/env bash
# BABILong launcher for one model/length combination on a single H20 GPU.
# Usage: run_babilong_single_h20.sh <model_path> <model_name> <gpu> <length> [tasks...]

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code}"
BABILONG_ROOT="${BABILONG_ROOT:-${PROJECT_ROOT}/babilong}"
MOM_ROOT="${MOM_ROOT:-${PROJECT_ROOT}/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
RESULTS_FOLDER="${RESULTS_FOLDER:-${MOM_ROOT}/babilong_results}"
DATASET_NAME="${DATASET_NAME:-RMT-team/babilong}"
LOGDIR="${LOGDIR:-${MOM_ROOT}/logs}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
TASKS_STR="${TASKS_STR:-}"

export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOGDIR" "$RESULTS_FOLDER"
cd "$BABILONG_ROOT"

MODEL_PATH=${1:?model_path required}
MODEL_NAME=${2:?model_name required}
GPU=${3:?gpu required}
LENGTH=${4:?length required}
shift 4

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
elif [[ -n "$TASKS_STR" ]]; then
  read -r -a TASKS <<< "$TASKS_STR"
else
  echo "No tasks provided. Pass tasks as args or TASKS_STR env." >&2
  exit 1
fi

LOGFILE="$LOGDIR/babilong_${MODEL_NAME}_${LENGTH}_$(date +%Y%m%d_%H%M).log"

EXTRA_FLAGS=()
case "$MODEL_NAME" in
  Llama-3.2-1B-Instruct|*Instruct*|*chat*)
    EXTRA_FLAGS=(--use_chat_template --use_instruction --use_examples --use_post_prompt)
    ;;
  *)
    EXTRA_FLAGS=(--use_instruction --use_examples --use_post_prompt)
    ;;
esac

{
  echo "============================================"
  echo "BABILong launcher"
  echo "  Model name: $MODEL_NAME"
  echo "  Model path: $MODEL_PATH"
  echo "  GPU: $GPU"
  echo "  Length: $LENGTH"
  echo "  Tasks: ${TASKS[*]}"
  echo "  Results: $RESULTS_FOLDER"
  echo "============================================"
} | tee -a "$LOGFILE"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/run_model_on_babilong.py \
  --results_folder "$RESULTS_FOLDER" \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$MODEL_PATH" \
  --tasks "${TASKS[@]}" \
  --lengths "$LENGTH" \
  "${EXTRA_FLAGS[@]}" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --api_url "" \
  2>&1 | tee -a "$LOGFILE"
