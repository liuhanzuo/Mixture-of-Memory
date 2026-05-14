#!/bin/bash
# H7 multi-ckpt BABILong eval — verify lambda_retrieve=0.1 fixes NIAH collapse.
# H7 is currently training on b200-4. We eval intermediate ckpts to see learning curve.

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
CKPT_DIR=${PROJECT_ROOT}/outputs/experiment_h7_low_niah

mkdir -p "$LOG_DIR"

# Run multiple H7 ckpts on different GPUs to characterize the fix's progression
declare -a JOBS=(
    "0|H7-step500|step_500.pt"
    "1|H7-step1000|step_1000.pt"
    "2|H7-step1500|step_1500.pt"
    "3|H7-step2000|step_2000.pt"
    "4|H7-step2500|step_2500.pt"
    "5|H7-step3000|step_3000.pt"
)

PIDS=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r GPU NAME CKPT <<< "$job"
    CKPT_FULL=${CKPT_DIR}/${CKPT}
    LOG=${LOG_DIR}/babilong_${NAME}_$(date +%Y%m%d_%H%M).log

    if [[ ! -f "$CKPT_FULL" ]]; then
        echo "SKIP: $CKPT_FULL not found"
        continue
    fi

    echo "[$(date)] Launching $NAME on GPU $GPU"

    # H7 uses dual-gate (same as H6) so include those flags
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
        --use_dual_gate \
        --forget_bias_init 1.0 \
        --input_bias_init 0.0 \
        --max_new_tokens 20 \
        --device cuda:0 \
        > "$LOG" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "  -> PID=$PID, log: $LOG"
    sleep 5
done

echo ""
echo "[$(date)] All ${#PIDS[@]} jobs launched. PIDs: ${PIDS[@]}"
