#!/bin/bash
# H-series v2 Phase 1 — Variant B: Joint Attention
# Slots prepended at every layer, joint self-attention
# Full fine-tune (backbone NOT frozen), 128 memory slots
#
# Usage: bash train_v2_B_joint.sh [NODE_NAME] [MASTER_PORT] [OUTPUT_DIR]

set -euo pipefail

NODE_NAME="${1:-local}"
MASTER_PORT="${2:-29501}"
OUTPUT_DIR="${3:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/h_v2_phase1_B_${NODE_NAME}}"
LOG_FILE="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/logs/h_v2_phase1_B_${NODE_NAME}.log"

PROJECT_ROOT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"
SCRIPT="${PROJECT_ROOT}/scripts/train_h_v2.py"
MODEL_PATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B"
DATASET_PATH="${PROJECT_ROOT}/data/armt_pg19_real_tokenized_full"

NUM_GPUS=$(nvidia-smi -L | wc -l)

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Starting Variant B training on ${NODE_NAME} with ${NUM_GPUS} GPUs"
echo "[$(date)] Output: ${OUTPUT_DIR}"
echo "[$(date)] Log: ${LOG_FILE}"

CMD="torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    ${SCRIPT} \
    --base_model ${MODEL_PATH} \
    --memory_variant B \
    --no_freeze_backbone \
    --num_slots 128 \
    --no_dual_gate \
    --dataset_path ${DATASET_PATH} \
    --segment_size 512 \
    --max_n_segments 2 \
    --no_loss_from_first_segment \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 5000 \
    --max_steps 50000 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --grad_clip 1.0 \
    --dtype bfloat16 \
    --output_dir ${OUTPUT_DIR} \
    --log_every 10 \
    --save_every 5000 \
    --seed 42"

# Launch in tmux to avoid SIGHUP
SESSION_NAME="v2_phase1_B_${NODE_NAME}"
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "cd ${PROJECT_ROOT} && ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "[$(date)] Launched in tmux session: ${SESSION_NAME}"
echo "  Attach: tmux attach -t ${SESSION_NAME}"
