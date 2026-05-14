#!/bin/bash
# Launch Beacon Qwen-2-7B-Instruct BABILong evaluation on an H20 node.
# Usage:
#   bash scripts/launch_h20_beacon.sh                 # default 6 length bins on GPUs 0-5
#   bash scripts/launch_h20_beacon.sh 1k:2 2k:3 ...  # custom length:gpu mapping

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code}"
WORKDIR="${WORKDIR:-${PROJECT_ROOT}/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
SCRIPT="${SCRIPT:-${WORKDIR}/scripts/eval_baseline_babilong.py}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/baselines/beacon-qwen-2-7b-hf}"
RESULTS="${RESULTS:-${WORKDIR}/babilong_results}"
OUTPUT_NAME="${OUTPUT_NAME:-Beacon-Qwen2-7B}"
LOGDIR="${LOGDIR:-${WORKDIR}/logs}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
TASKS_STR="${TASKS_STR:-qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/hf_cache/datasets}"

export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE

mkdir -p "$HF_DATASETS_CACHE" "$LOGDIR" "$RESULTS"
cd "$WORKDIR"
read -r -a TASKS <<< "$TASKS_STR"

if [[ $# -gt 0 ]]; then
  LENGTHS_GPU=("$@")
else
  LENGTHS_GPU=(
    "1k:0"
    "2k:1"
    "4k:2"
    "8k:3"
    "16k:4"
    "32k:5"
  )
fi

for entry in "${LENGTHS_GPU[@]}"; do
  L="${entry%%:*}"
  G="${entry##*:}"
  LOG="$LOGDIR/babilong_beacon_${L}_$(date +%Y%m%d_%H%M).log"
  echo "[launch] gpu=$G length=$L log=$LOG"
  CUDA_VISIBLE_DEVICES="$G" nohup "$PYTHON_BIN" "$SCRIPT" \
    --baseline beacon \
    --model_path "$MODEL_PATH" \
    --output_name "$OUTPUT_NAME" \
    --results_folder "$RESULTS" \
    --tasks "${TASKS[@]}" \
    --lengths "$L" \
    --use_chat_template --use_instruction --use_examples --use_post_prompt \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --limit "$LIMIT" \
    > "$LOG" 2>&1 &
  disown
  sleep 1
done

echo "Launched ${#LENGTHS_GPU[@]} Beacon jobs"
ps -ef | grep eval_baseline_babilong | grep -v grep | wc -l
