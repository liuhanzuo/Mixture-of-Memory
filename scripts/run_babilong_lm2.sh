#!/bin/bash
# BABILong evaluation for LM2 (Large Memory Model)
# Launches single-GPU evaluation on all qa1-qa5 tasks across 0k-32k lengths.

set -euo pipefail

# Proxy for HuggingFace dataset access
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# LM2 source on PYTHONPATH
export PYTHONPATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/LM2:${PYTHONPATH:-}

# Single GPU
export CUDA_VISIBLE_DEVICES=0

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON=/opt/conda/envs/torch-base/bin/python
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

LOGFILE="${LOG_DIR}/babilong_lm2_iter12000_$(date +%Y%m%d_%H%M).log"

echo "[$(date)] Starting BABILong LM2 evaluation" | tee "$LOGFILE"
echo "[$(date)] Log: $LOGFILE" | tee -a "$LOGFILE"

$PYTHON "${SCRIPT_DIR}/run_babilong_lm2.py" \
    --ckpt_path "${PROJECT_ROOT}/outputs/lm2_b200_4/ckpts_20260509_152240/ckpt_iter_12000.pth" \
    --model_name "/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B" \
    --results_folder "${PROJECT_ROOT}/babilong_results" \
    --results_name "LM2-iter12000" \
    --tasks qa1 qa2 qa3 qa4 qa5 \
    --lengths 0k 1k 2k 4k 8k 16k 32k \
    --dataset_name "RMT-team/babilong" \
    --chunk_size 2048 \
    --memory_slots 2048 \
    --max_new_tokens 20 \
    --device cuda:0 \
    2>&1 | tee -a "$LOGFILE"

echo "[$(date)] BABILong LM2 evaluation finished" | tee -a "$LOGFILE"
