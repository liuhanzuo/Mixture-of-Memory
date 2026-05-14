#!/bin/bash
# BABILong evaluation for MemoryLLM-8B-chat

set -euo pipefail

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONPATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/MemoryLLM-source:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
LOG_DIR=${PROJECT_ROOT}/logs
SCRIPT=${PROJECT_ROOT}/scripts/run_babilong_memoryllm.py
PYTHON=/opt/conda/envs/torch-base/bin/python

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/babilong_memoryllm_8b_chat_$(date +%Y%m%d_%H%M).log"

GPU=${1:-7}
echo "[$(date)] Launching MemoryLLM-8B-chat BABILong eval on GPU $GPU"
echo "Log: $LOGFILE"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT" \
    --model_path /apdcephfs_wzc1/share_303098609/pighzliu_code/baselines/memoryllm-8b-chat \
    --output_name MemoryLLM-8B-chat \
    --tasks qa1 qa2 qa3 qa4 qa5 \
    --lengths 0k 1k 2k 4k 8k 16k 32k \
    --max_new_tokens 20 \
    --device cuda:0 \
    2>&1 | tee "$LOGFILE"
