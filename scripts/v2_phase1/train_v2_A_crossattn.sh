#!/bin/bash
# H-series v2 Phase 1 — Variant A: Cross-Attention Slots
# Write at layer 8, read at layers 10/12/14 (adapted for Llama-3.2-1B with 16 layers)
# Frozen backbone, trainable memory modules only
#
# Usage: bash train_v2_A_crossattn.sh [NODE_NAME] [MASTER_PORT] [OUTPUT_DIR]

set -euo pipefail

NODE_NAME="${1:-local}"
MASTER_PORT="${2:-29500}"
OUTPUT_DIR="${3:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/h_v2_phase1_A_${NODE_NAME}}"
LOG_FILE="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/logs/h_v2_phase1_A_${NODE_NAME}.log"

PROJECT_ROOT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"
SCRIPT="${PROJECT_ROOT}/scripts/train_h_v2.py"
MODEL_PATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B"
DATASET_PATH="${PROJECT_ROOT}/data/armt_pg19_real_tokenized_full"

NUM_GPUS=$(nvidia-smi -L | wc -l)

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Starting Variant A training on ${NODE_NAME} with ${NUM_GPUS} GPUs"
echo "[$(date)] Output: ${OUTPUT_DIR}"
echo "[$(date)] Log: ${LOG_FILE}"

CMD="torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    ${SCRIPT} \
    --base_model ${MODEL_PATH} \
    --memory_variant A \
    --freeze_backbone \
    --num_slots 64 \
    --memory_write_layer 8 \
    --memory_read_layers 10,12,14 \
    --write_lr 0.1 \
    --residual_scale 0.01 \
    --use_dual_gate \
    --forget_bias_init 1.0 \
    --input_bias_init 0.0 \
    --dataset_path ${DATASET_PATH} \
    --segment_size 512 \
    --max_n_segments 2 \
    --no_loss_from_first_segment \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 5000 \
    --max_steps 50000 \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --grad_clip 1.0 \
    --dtype bfloat16 \
    --output_dir ${OUTPUT_DIR} \
    --log_every 10 \
    --save_every 5000 \
    --seed 42"

# Launch in tmux to avoid SIGHUP
SESSION_NAME="v2_phase1_A_${NODE_NAME}"
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "cd ${PROJECT_ROOT} && ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "[$(date)] Launched in tmux session: ${SESSION_NAME}"
echo "  Attach: tmux attach -t ${SESSION_NAME}"
