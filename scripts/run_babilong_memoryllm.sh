#!/bin/bash
# BABILong evaluation for MemoryLLM-8B-chat
#
# IMPORTANT (2026-06-23): MemoryLLM-source's custom forward is only numerically
# correct under transformers 4.43.x. The project's default envs (transformers 5.x)
# make the model emit degenerate token-0 garbage ("!!!!" / "MarcusMarcus...").
# The pinned requirements_infer_only.txt torch (2.5.1+cu121) SIGFPEs the PEFT LoRA
# GEMM on H20 (sm90). The working combo, installed in external/memoryllm_venv, is:
#   transformers==4.43.4 + peft==0.10.0 + torch==2.6.0+cu124
# Verified: qa1/0k 5-sample smoke = 2/5 correct, well-formatted, non-repetitive.

set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}
MEMORYLLM_SRC=${MEMORYLLM_SRC:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/MemoryLLM-source}
export PYTHONPATH=${MEMORYLLM_SRC}:${PYTHONPATH:-}
LOG_DIR=${PROJECT_ROOT}/logs
SCRIPT=${PROJECT_ROOT}/scripts/run_babilong_memoryllm.py
# Pinned-stack venv (transformers 4.43.4 / peft 0.10.0 / torch 2.6.0+cu124).
PYTHON=${PYTHON_BIN:-${PROJECT_ROOT}/external/memoryllm_venv/bin/python}
# Locally cached HF snapshot of YuWangX/memoryllm-8b-chat.
MODEL_PATH=${MODEL_PATH:-$(ls -d ${PROJECT_ROOT}/.hf_cache/models--YuWangX--memoryllm-8b-chat/snapshots/*/ | head -1)}

export HF_HOME=${HF_HOME:-${PROJECT_ROOT}/.hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${PROJECT_ROOT}/.hf_cache/datasets}

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/babilong_memoryllm_8b_chat_$(date +%Y%m%d_%H%M).log"

GPU=${1:-7}
echo "[$(date)] Launching MemoryLLM-8B-chat BABILong eval on GPU $GPU"
echo "Python: $PYTHON"
echo "Model:  $MODEL_PATH"
echo "Log: $LOGFILE"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --output_name MemoryLLM-8B-chat \
    --tasks qa1 qa2 qa3 qa4 qa5 \
    --lengths 0k 1k 2k 4k 8k 16k 32k \
    --max_new_tokens 20 \
    --device cuda:0 \
    2>&1 | tee "$LOGFILE"
