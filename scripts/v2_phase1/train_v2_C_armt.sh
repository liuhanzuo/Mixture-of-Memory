#!/bin/bash
# H-series v2 Phase 1 — Variant C: ARMT-style prepend memory tokens
# Uses ARMT's run_finetuning_lm_rmt_hf.py with Llama-3.2-1B
# Frozen backbone, trainable ARMT memory cells
#
# Usage: bash train_v2_C_armt.sh [NODE_NAME] [MASTER_PORT] [OUTPUT_DIR]

set -euo pipefail

NODE_NAME="${1:-local}"
MASTER_PORT="${2:-29502}"
OUTPUT_DIR="${3:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/h_v2_phase1_C_${NODE_NAME}}"
LOG_FILE="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/logs/h_v2_phase1_C_${NODE_NAME}.log"

PROJECT_ROOT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
TORCHRUN_CMD="${TORCHRUN_CMD:-${PYTHON_BIN} -m torch.distributed.run}"
ARMT_DIR="${PROJECT_ROOT}/third_party/associative-recurrent-memory-transformer"
SCRIPT="${ARMT_DIR}/run_finetuning_lm_rmt_hf.py"
MODEL_PATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B"
DATASET_PATH="${PROJECT_ROOT}/data/armt_pg19_real_tokenized_full"

NUM_GPUS=$(nvidia-smi -L | wc -l)

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Starting Variant C (ARMT) training on ${NODE_NAME} with ${NUM_GPUS} GPUs"
echo "[$(date)] Output: ${OUTPUT_DIR}"
echo "[$(date)] Log: ${LOG_FILE}"

# ARMT uses HuggingFace Trainer + accelerate
CMD="cd ${ARMT_DIR} && ${TORCHRUN_CMD} \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    ${SCRIPT} \
    --from_pretrained ${MODEL_PATH} \
    --tokenizer ${MODEL_PATH} \
    --model_cls transformers:AutoModelForCausalLM \
    --model_type decoder \
    --memory_cell_cls modeling_amt.online_armt:AssociativeMemoryCell \
    --recurrent_wrapper_cls modeling_amt.online_armt:AssociativeRecurrentWrapper \
    --tokenized_dataset ${DATASET_PATH} \
    --train_tokens tokens \
    --valid_tokens tokens \
    --task_name pg19 \
    --segment_size 512 \
    --max_n_segments 2 \
    --sample_size 1024 \
    --num_mem_tokens 32 \
    --d_mem 64 \
    --no_loss_from_first_segment \
    --freeze_model_weights \
    --bptt_depth 2 \
    --attn_implementation sdpa \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_steps 5000 \
    --max_steps 50000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 10 \
    --save_steps 5000 \
    --seed 42 \
    --dataloader_num_workers 2"

# Launch in tmux to avoid SIGHUP
SESSION_NAME="v2_phase1_C_${NODE_NAME}"
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "${CMD} 2>&1 | tee ${LOG_FILE}"

echo "[$(date)] Launched in tmux session: ${SESSION_NAME}"
echo "  Attach: tmux attach -t ${SESSION_NAME}"
