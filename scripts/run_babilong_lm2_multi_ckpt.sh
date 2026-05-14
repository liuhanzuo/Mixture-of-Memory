#!/bin/bash
# BABILong-100 parallel evaluation for multiple LM2 checkpoints.
# Runs each ckpt on a separate GPU. Default: 4 ckpts on GPUs 0-3.
#
# Each iter takes ~37 min on a single L20A.

set -euo pipefail

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONPATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/LM2:${PYTHONPATH:-}

PROJECT_ROOT=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
LOG_DIR=${PROJECT_ROOT}/logs
SCRIPT_DIR=${PROJECT_ROOT}/scripts
PYTHON=/opt/conda/envs/torch-base/bin/python
CKPT_DIR=${PROJECT_ROOT}/outputs/lm2_b200_4/ckpts_20260509_152240

mkdir -p "$LOG_DIR"

# (gpu_id, iter_num) pairs
declare -a JOBS=(
    "0 8000"
    "1 10000"
    "2 14000"
    "3 16000"
)

PIDS=()
for job in "${JOBS[@]}"; do
    GPU=${job%% *}
    ITER=${job##* }
    CKPT=${CKPT_DIR}/ckpt_iter_${ITER}.pth
    NAME=LM2-iter${ITER}
    LOG=${LOG_DIR}/babilong_${NAME}_$(date +%Y%m%d_%H%M).log

    if [[ ! -f "$CKPT" ]]; then
        echo "SKIP: ckpt not found: $CKPT"
        continue
    fi

    echo "[$(date)] Launching $NAME on GPU $GPU -> log $LOG"

    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON "${SCRIPT_DIR}/run_babilong_lm2.py" \
        --ckpt_path "$CKPT" \
        --model_name "/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B" \
        --results_folder "${PROJECT_ROOT}/babilong_results" \
        --results_name "$NAME" \
        --tasks qa1 qa2 qa3 qa4 qa5 \
        --lengths 0k 1k 2k 4k 8k 16k 32k \
        --dataset_name "RMT-team/babilong" \
        --chunk_size 2048 \
        --memory_slots 2048 \
        --max_new_tokens 20 \
        --device cuda:0 \
        > "$LOG" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "  -> PID=$PID"
    sleep 5  # stagger startup so HF dataset cache writes don't collide
done

echo ""
echo "[$(date)] All ${#PIDS[@]} jobs launched. PIDs: ${PIDS[@]}"
echo "Logs in: $LOG_DIR/babilong_LM2-iter*.log"
