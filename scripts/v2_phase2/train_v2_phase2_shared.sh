#!/bin/bash
# Shared H-series v2 Phase 2 launcher for BABILong qa1 curriculum.
# Usage: bash train_v2_phase2_shared.sh <A|B|D> <NODE_NAME> [MASTER_PORT] <PHASE1_CKPT> [OUTPUT_DIR]

set -euo pipefail

VARIANT="${1:?variant A/B/D required}"
NODE_NAME="${2:?node name required}"
MASTER_PORT="${3:-29610}"
PHASE1_CKPT="${4:?phase1 checkpoint required}"

PROJECT_ROOT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"
SCRIPT="${PROJECT_ROOT}/scripts/train_h_v2_babilong.py"
MODEL_PATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B"
NOISE_DATASET_PATH="${PROJECT_ROOT}/data/armt_pg19_real_tokenized_full"
BABI_DATA_ROOT="/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong/data"
BABI_PATH="${BABI_DATA_ROOT}/tasks_1-20_v1-2/en-10k"
ZIP_PATH="${BABI_DATA_ROOT}/tasks_1-20_v1-2.zip"
OUTPUT_DIR="${5:-${PROJECT_ROOT}/outputs/h_v2_phase2_${VARIANT}_${NODE_NAME}}"
LOG_FILE="${PROJECT_ROOT}/logs/h_v2_phase2_${VARIANT}_${NODE_NAME}.log"
SESSION_NAME="v2_phase2_${VARIANT}_${NODE_NAME}"
NUM_GPUS=$(nvidia-smi -L | wc -l)

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing training script: ${SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "Missing phase1 checkpoint: ${PHASE1_CKPT}" >&2
  exit 1
fi

if [[ ! -d "${BABI_PATH}" ]]; then
  if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "Missing bAbI zip: ${ZIP_PATH}" >&2
    exit 1
  fi
  echo "[$(date)] Extracting bAbI tasks into ${BABI_DATA_ROOT}"
  unzip -q -o "${ZIP_PATH}" -d "${BABI_DATA_ROOT}"
fi

case "${VARIANT}" in
  A)
    VARIANT_ARGS="--memory_variant A --freeze_backbone --num_slots 64 --memory_write_layer 8 --memory_read_layers 10,12,14 --write_lr 0.1 --residual_scale 0.01 --use_dual_gate --forget_bias_init 1.0 --input_bias_init 0.0"
    ;;
  B)
    VARIANT_ARGS="--memory_variant B --no_freeze_backbone --num_slots 128 --no_dual_gate"
    ;;
  D)
    VARIANT_ARGS="--memory_variant D --freeze_backbone --num_slots 64 --memory_write_layer 8 --memory_read_layers 10,12,14 --write_lr 0.1 --residual_scale 0.01 --use_dual_gate --forget_bias_init 1.0 --input_bias_init 0.0 --lora_r 8 --lora_alpha 32"
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

CURRICULUM=(2 4 8 16 32)

RUN_CMD="set -euo pipefail
cd \"${PROJECT_ROOT}\"
CURRENT_CKPT=\"${PHASE1_CKPT}\"
echo \"[\$(date)] Starting H-v2 Phase 2 variant ${VARIANT} on ${NODE_NAME}\"
for N_SEG in ${CURRICULUM[*]}; do
  STAGE_DIR=\"${OUTPUT_DIR}/segments_\${N_SEG}\"
  mkdir -p \"\${STAGE_DIR}\"
  echo \"[\$(date)] Phase 2 ${VARIANT}: launching curriculum stage N_SEG=\${N_SEG} from ckpt \${CURRENT_CKPT}\"
  torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} ${SCRIPT} --base_model ${MODEL_PATH} ${VARIANT_ARGS} --noise_dataset_path ${NOISE_DATASET_PATH} --babi_path ${BABI_PATH} --task_dataset qa1_single-supporting-fact --segment_size 512 --max_n_segments \"\${N_SEG}\" --lr 1e-4 --weight_decay 0.01 --warmup_steps 1000 --max_steps 6000 --batch_size 1 --gradient_accumulation_steps 64 --grad_clip 1.0 --dtype bfloat16 --output_dir \"\${STAGE_DIR}\" --log_every 10 --save_every 1000 --seed 42 --resume_checkpoint \"\${CURRENT_CKPT}\"
  CURRENT_CKPT=\"\${STAGE_DIR}/checkpoint_final.pt\"
done"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${RUN_CMD}") 2>&1 | tee $(printf '%q' "${LOG_FILE}")"

echo "[$(date)] Launched in tmux session: ${SESSION_NAME}"
echo "  Attach: tmux attach -t ${SESSION_NAME}"
echo "  Log: ${LOG_FILE}"
