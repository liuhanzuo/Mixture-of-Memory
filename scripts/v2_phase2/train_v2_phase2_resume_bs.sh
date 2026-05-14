#!/bin/bash
# H-series v2 Phase 2 launcher with per-variant bs/grad_accum and resume-from-ckpt.
#
# Difference from train_v2_phase2_shared.sh:
#   - Accepts START_N_SEG to resume curriculum at a specific segment count
#     (instead of always restarting from segments=2).
#   - Accepts a RESUME_CKPT that can point to any segments_N/checkpoint_K.pt.
#   - Per-variant BS/GRAD_ACCUM keeps effective_batch = 512 across all variants
#     (BS * GRAD_ACCUM * NUM_GPUS == 512), but uses larger micro-batch to amortize
#     the forward/backward cost.
#
# Usage:
#   bash train_v2_phase2_resume_bs.sh <A|B|D> <NODE_NAME> <MASTER_PORT> <RESUME_CKPT> <START_N_SEG> [OUTPUT_DIR]
#
# Example (A from segments_2/ckpt_3000):
#   bash scripts/v2_phase2/train_v2_phase2_resume_bs.sh A b2001 29610 \
#       /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/h_v2_phase2_A_b2001/segments_2/checkpoint_3000.pt 2

set -euo pipefail

VARIANT="${1:?variant A/B/D required}"
NODE_NAME="${2:?node name required}"
MASTER_PORT="${3:-29610}"
RESUME_CKPT="${4:?resume checkpoint path required}"
START_N_SEG="${5:?start curriculum segment count required (e.g. 2)}"

PROJECT_ROOT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"
SCRIPT="${PROJECT_ROOT}/scripts/train_h_v2_babilong.py"
MODEL_PATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B"
NOISE_DATASET_PATH="${PROJECT_ROOT}/data/armt_pg19_real_tokenized_full"
BABI_DATA_ROOT="/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong/data"
BABI_PATH="${BABI_DATA_ROOT}/tasks_1-20_v1-2/en-10k"
ZIP_PATH="${BABI_DATA_ROOT}/tasks_1-20_v1-2.zip"
OUTPUT_DIR="${6:-${PROJECT_ROOT}/outputs/h_v2_phase2_${VARIANT}_${NODE_NAME}}"
LOG_FILE="${PROJECT_ROOT}/logs/h_v2_phase2_${VARIANT}_${NODE_NAME}_bs.log"
SESSION_NAME="v2_phase2_${VARIANT}_${NODE_NAME}_bs"
NUM_GPUS=$(nvidia-smi -L | wc -l)

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing training script: ${SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CKPT}" >&2
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

# --------------------------------------------------------------------------- #
# Per-variant architecture args + per-variant bs/grad_accum.
#
# Effective batch invariant: BS * GRAD_ACCUM * NUM_GPUS == 512
# (assumes NUM_GPUS == 8). Adjust manually if NUM_GPUS differs.
# --------------------------------------------------------------------------- #
case "${VARIANT}" in
  A)
    VARIANT_ARGS="--memory_variant A --freeze_backbone --num_slots 64 --memory_write_layer 8 --memory_read_layers 10,12,14 --write_lr 0.1 --residual_scale 0.01 --use_dual_gate --forget_bias_init 1.0 --input_bias_init 0.0"
    BS=32
    GRAD_ACCUM=2
    ;;
  B)
    # B unfreezes the backbone (full 1B finetune) so we keep micro-batch smaller.
    VARIANT_ARGS="--memory_variant B --no_freeze_backbone --num_slots 128 --no_dual_gate"
    BS=16
    GRAD_ACCUM=4
    ;;
  D)
    VARIANT_ARGS="--memory_variant D --freeze_backbone --num_slots 64 --memory_write_layer 8 --memory_read_layers 10,12,14 --write_lr 0.1 --residual_scale 0.01 --use_dual_gate --forget_bias_init 1.0 --input_bias_init 0.0 --lora_r 8 --lora_alpha 32"
    BS=32
    GRAD_ACCUM=2
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

# Sanity check: warn if effective batch deviates from the original 512.
EFFECTIVE_BATCH=$(( BS * GRAD_ACCUM * NUM_GPUS ))
if [[ "${EFFECTIVE_BATCH}" -ne 512 ]]; then
  echo "[WARN] effective batch = ${EFFECTIVE_BATCH} (BS=${BS} * GA=${GRAD_ACCUM} * GPUS=${NUM_GPUS}); original was 512" >&2
fi

# --------------------------------------------------------------------------- #
# Curriculum: 2 -> 4 -> 8 -> 16 -> 32, but skip stages before START_N_SEG.
# We resume the FIRST (current) stage from the supplied checkpoint, and let
# subsequent stages chain via checkpoint_final.pt as before.
# --------------------------------------------------------------------------- #
ALL_CURRICULUM=(2 4 8 16 32)
CURRICULUM=()
SEEN_START=0
for n in "${ALL_CURRICULUM[@]}"; do
  if [[ "${n}" == "${START_N_SEG}" ]]; then
    SEEN_START=1
  fi
  if [[ "${SEEN_START}" -eq 1 ]]; then
    CURRICULUM+=("${n}")
  fi
done

if [[ "${#CURRICULUM[@]}" -eq 0 ]]; then
  echo "START_N_SEG=${START_N_SEG} not in curriculum (${ALL_CURRICULUM[*]})" >&2
  exit 1
fi

RUN_CMD="set -euo pipefail
cd \"${PROJECT_ROOT}\"
CURRENT_CKPT=\"${RESUME_CKPT}\"
echo \"[\$(date)] Starting H-v2 Phase 2 variant ${VARIANT} on ${NODE_NAME} (BS=${BS} GA=${GRAD_ACCUM} effective=${EFFECTIVE_BATCH})\"
for N_SEG in ${CURRICULUM[*]}; do
  STAGE_DIR=\"${OUTPUT_DIR}/segments_\${N_SEG}\"
  mkdir -p \"\${STAGE_DIR}\"
  echo \"[\$(date)] Phase 2 ${VARIANT}: launching curriculum stage N_SEG=\${N_SEG} from ckpt \${CURRENT_CKPT}\"
  torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} ${SCRIPT} --base_model ${MODEL_PATH} ${VARIANT_ARGS} --noise_dataset_path ${NOISE_DATASET_PATH} --babi_path ${BABI_PATH} --task_dataset qa1_single-supporting-fact --segment_size 512 --max_n_segments \"\${N_SEG}\" --lr 1e-4 --weight_decay 0.01 --warmup_steps 1000 --max_steps 6000 --batch_size ${BS} --gradient_accumulation_steps ${GRAD_ACCUM} --grad_clip 1.0 --dtype bfloat16 --output_dir \"\${STAGE_DIR}\" --log_every 10 --save_every 1000 --seed 42 --resume_checkpoint \"\${CURRENT_CKPT}\"
  CURRENT_CKPT=\"\${STAGE_DIR}/checkpoint_final.pt\"
done"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${RUN_CMD}") 2>&1 | tee $(printf '%q' "${LOG_FILE}")"

echo "[$(date)] Launched in tmux session: ${SESSION_NAME}"
echo "  Variant: ${VARIANT}, BS=${BS}, GRAD_ACCUM=${GRAD_ACCUM}, effective=${EFFECTIVE_BATCH}"
echo "  Resume from: ${RESUME_CKPT}"
echo "  Curriculum from: ${CURRICULUM[*]}"
echo "  Attach: tmux attach -t ${SESSION_NAME}"
echo "  Log: ${LOG_FILE}"
