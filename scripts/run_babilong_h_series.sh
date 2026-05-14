#!/bin/bash
# BABILong-100 parallel evaluation for H1-H5 series ckpts (cross-attention memory).
# Each ckpt runs on a separate GPU on b200-2.
# H6/H6b are already running on GPU 5/6, so we use GPU 0,1,2,3,7.

set -euo pipefail

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONPATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
LOG_DIR=${PROJECT_ROOT}/logs
SCRIPT=${PROJECT_ROOT}/scripts/run_babilong_h6.py
PYTHON=/opt/conda/envs/torch-base/bin/python
MODEL_PATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b

mkdir -p "$LOG_DIR"

# (gpu_id, output_name, ckpt_path, extra_args) tuples
# H series — all use 64 slots, write@16, read@18,22,26,30, init=strided
# H/H2/H3/H4/H5/H5b have NO dual-gate (predates H6)
declare -a JOBS=(
    "0|H-step5000|outputs/experiment_h_middle_layer/step_5000.pt|"
    "1|H2-step5000|outputs/experiment_h2_deeper/step_5000.pt|--memory_write_layer 20 --memory_read_layers 22,25,28,31"
    "2|H3-step5000|outputs/experiment_h3_aggressive_niah/step_5000.pt|--num_slots 128"
    "3|H4-step3000|outputs/experiment_h4_early_write/step_3000.pt|--memory_write_layer 12"
    "4|H5-step2000|outputs/experiment_h5_fixed/step_2000.pt|"
    "7|H5b-step2000|outputs/experiment_h5b_fixed/step_2000.pt|"
)

PIDS=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r GPU NAME CKPT EXTRA <<< "$job"
    CKPT_FULL=${PROJECT_ROOT}/${CKPT}
    LOG=${LOG_DIR}/babilong_${NAME}_$(date +%Y%m%d_%H%M).log

    if [[ ! -f "$CKPT_FULL" ]]; then
        echo "SKIP: $CKPT_FULL not found"
        continue
    fi

    echo "[$(date)] Launching $NAME on GPU $GPU"
    echo "  ckpt: $CKPT_FULL ($(stat -c %s $CKPT_FULL | numfmt --to=iec))"

    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON $SCRIPT \
        --ckpt_path "$CKPT_FULL" \
        --model_path "$MODEL_PATH" \
        --output_name "$NAME" \
        --tasks qa1 qa2 qa3 qa4 qa5 \
        --lengths 0k 1k 2k 4k 8k 16k 32k \
        --chunk_size 4096 \
        --num_slots 64 \
        --memory_write_layer 16 \
        --memory_read_layers "18,22,26,30" \
        --memory_init strided \
        --max_new_tokens 20 \
        --device cuda:0 \
        $EXTRA \
        > "$LOG" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "  -> PID=$PID, log: $LOG"
    sleep 5
done

echo ""
echo "[$(date)] All ${#PIDS[@]} jobs launched. PIDs: ${PIDS[@]}"
