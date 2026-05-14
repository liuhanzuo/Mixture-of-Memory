#!/bin/bash
# ARMT full PG19 training relaunch with fixed save strategy.
#
# Based on upstream scripts/pg19/finetune_armt_llama3.2_pg19_sliding.sh, adapted to:
#   - use local cephfs share paths (no meta-llama HF download at train time)
#   - save checkpoints every save_steps
#   - default to a larger batch than the first recovery run: bs=16 per-GPU × grad_accum=1 × 8 GPU = 128
#   - allow env overrides for batch/env selection during future recoveries
#   - output_dir -> outputs/armt_pg19_full_${NODE} (absolute path, no cwd ambiguity)
#
# Usage:
#   bash scripts/train_armt_pg19_full_v2.sh <NODE_NAME> [MASTER_PORT]
# Example on b200-5 (ephemeral):
#   bash scripts/train_armt_pg19_full_v2.sh b2005

set -euo pipefail

NODE_NAME="${1:?node name required, e.g. b2005}"
MASTER_PORT="${2:-29002}"

# Project on ephemeral share (b200-5..8 mount share_304376610 at /apdcephfs_wzc1)
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
ARMT_ROOT="${PROJECT_ROOT}/third_party/associative-recurrent-memory-transformer"
TOKENIZED_DATASET="${PROJECT_ROOT}/data/armt_pg19_real_tokenized_full"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B}"
# fallback model path if share_303 is not visible on ephemeral nodes (b200-5..8 see share_304)
if [[ ! -d "${MODEL_PATH}" ]]; then
  MODEL_PATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Llama-3.2-1B"
fi

OUTPUT_DIR="${PROJECT_ROOT}/outputs/armt_pg19_full_${NODE_NAME}"
LOG_FILE="${PROJECT_ROOT}/logs/armt_pg19_full_${NODE_NAME}_$(date +%Y%m%d_%H%M).log"
SESSION_NAME="armt_full_${NODE_NAME}"
NUM_GPUS=$(nvidia-smi -L | wc -l)

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${OUTPUT_DIR}"

# ----- ARMT hyperparameters (upstream-consistent) -----
MODEL_TYPE=decoder
MEMORY_CELL="modeling_amt.language_modeling:AssociativeMemoryCell"
RECURRENT_WRAPPER="modeling_amt.language_modeling:AssociativeRecurrentWrapper"
BACKBONE_CLS="transformers:AutoModelForCausalLM"
DATASET_NAME=pg19

ITERS=50000
TBS="${TBS:-128}"                # bigger than the previous recovery run's 64
BS="${BS:-16}"                   # per-GPU micro-batch; user requested bigger bs this round
GRAD_ACC="${GRAD_ACC:-$(( TBS / (BS * NUM_GPUS) ))}"

if (( BS * NUM_GPUS * GRAD_ACC != TBS )); then
  echo "[ERROR] effective batch mismatch: BS=$BS NUM_GPUS=$NUM_GPUS GA=$GRAD_ACC != TBS=$TBS" >&2
  exit 1
fi

LR=1e-05
SEGMENT_SIZE=512
MAX_N_SEGMENTS=2
MEMORY_SIZE=32
D_MEM=64
LAYERS_ATTR=model.layers
SAMPLE_SIZE=$(( MAX_N_SEGMENTS * SEGMENT_SIZE ))
SCHEDULER=linear
WARMUP=$(( ITERS / 10 ))
K2=-1

# Save strategy (THE FIX)
SAVE_STEPS=2500                  # 20 ckpts over 50k iters + final
SAVE_TOTAL_LIMIT=5               # keep last 5 + best

ACCEL_CONFIG="${ARMT_ROOT}/accel_configs/accelerate_bf16.yaml"
if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "[ERROR] accelerate config not found: ${ACCEL_CONFIG}" >&2
  exit 1
fi

# Prefer an sm100-capable torch env first. Returned replacement B200 nodes may expose
# L20A / sm_100 where the older MemLong env fails with "no kernel image".
TORCH_BASE_SM100_PYTHON="${TORCH_BASE_SM100_PYTHON:-/opt/conda/envs/torch-base-sm100/bin/python}"
TORCH_BASE_PYTHON="${TORCH_BASE_PYTHON:-/opt/conda/envs/torch-base/bin/python}"
MEMLONG_PYTHON="${PROJECT_ROOT}/MemLong/memlong_env/bin/python"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ -x "${TORCH_BASE_SM100_PYTHON}" ]] && "${TORCH_BASE_SM100_PYTHON}" -c "import torch, transformers, accelerate, fla" 2>/dev/null; then
  ACCELERATE_CMD="${TORCH_BASE_SM100_PYTHON} -m accelerate.commands.launch"
elif [[ -x "${TORCH_BASE_PYTHON}" ]] && "${TORCH_BASE_PYTHON}" -c "import torch, transformers, accelerate, fla" 2>/dev/null; then
  ACCELERATE_CMD="${TORCH_BASE_PYTHON} -m accelerate.commands.launch"
elif [[ -x "${VENV_PYTHON}" ]] && "${VENV_PYTHON}" -c "import torch, transformers, accelerate, fla" 2>/dev/null; then
  ACCELERATE_CMD="${VENV_PYTHON} -m accelerate.commands.launch"
elif [[ -x "${MEMLONG_PYTHON}" ]] && "${MEMLONG_PYTHON}" -c "import torch, transformers, accelerate" 2>/dev/null; then
  ACCELERATE_CMD="${MEMLONG_PYTHON} -m accelerate.commands.launch"
else
  ACCELERATE_CMD="accelerate launch"  # fallback to PATH
fi

if [[ ! -d "${TOKENIZED_DATASET}" ]]; then
  echo "[ERROR] tokenized dataset not found: ${TOKENIZED_DATASET}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] base model not found: ${MODEL_PATH}" >&2
  exit 1
fi

echo "=== ARMT full relaunch ==="
echo "  NODE=${NODE_NAME}   GPUS=${NUM_GPUS}"
echo "  TBS=${TBS}   BS=${BS}   GRAD_ACC=${GRAD_ACC}   (effective_batch=$(( BS * NUM_GPUS * GRAD_ACC )))"
echo "  ITERS=${ITERS}   WARMUP=${WARMUP}   SAVE_STEPS=${SAVE_STEPS}   SAVE_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  LOG=${LOG_FILE}"

RUN_CMD="set -euo pipefail
cd \"${ARMT_ROOT}\"
echo \"[\$(date)] launching ARMT full on ${NODE_NAME} using ${ACCELERATE_CMD}\"
${ACCELERATE_CMD} \\
    --config_file ${ACCEL_CONFIG} \\
    --main_process_port ${MASTER_PORT} \\
    --num_processes ${NUM_GPUS} \\
    --mixed_precision bf16 \\
    run_finetuning_lm_rmt_hf.py \\
    --tokenized_dataset ${TOKENIZED_DATASET} \\
    --output_dir ${OUTPUT_DIR} \\
    --from_pretrained ${MODEL_PATH} \\
    --model_type ${MODEL_TYPE} \\
    --memory_cell_cls ${MEMORY_CELL} \\
    --recurrent_wrapper_cls ${RECURRENT_WRAPPER} \\
    --model_cls ${BACKBONE_CLS} \\
    --segment_size ${SEGMENT_SIZE} \\
    --sample_size ${SAMPLE_SIZE} \\
    --val_sample_size ${SAMPLE_SIZE} \\
    --num_mem_tokens ${MEMORY_SIZE} \\
    --max_n_segments ${MAX_N_SEGMENTS} \\
    --min_sample_len 16000 \\
    --per_device_train_batch_size ${BS} \\
    --gradient_accumulation_steps ${GRAD_ACC} \\
    --max_steps ${ITERS} \\
    --metric_for_best_model eval_loss \\
    --greater_is_better false \\
    --save_strategy steps \\
    --save_steps ${SAVE_STEPS} \\
    --save_total_limit ${SAVE_TOTAL_LIMIT} \\
    --load_best_model_at_end \\
    --k2 ${K2} \\
    --optimizer AdamW \\
    --weight_decay 0.01 \\
    --learning_rate ${LR} \\
    --lr_scheduler_type ${SCHEDULER} \\
    --warmup_steps ${WARMUP} \\
    --data_n_workers 2 \\
    --logging_steps 25 \\
    --eval_steps ${SAVE_STEPS} \\
    --show_valid_examples 5 \\
    --seed 42 \\
    --d_mem ${D_MEM} \\
    --layers_attr ${LAYERS_ATTR} \\
    --no_loss_from_first_segment \\
    --valid_tokens tokens \\
    --train_tokens tokens \\
    --prev_seg_kv \\
    --use_sink \\
    --attn_implementation eager
echo \"[\$(date)] ARMT full on ${NODE_NAME} EXITED\"
"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${RUN_CMD}") 2>&1 | tee $(printf '%q' "${LOG_FILE}")"

echo "[$(date)] Launched in tmux session: ${SESSION_NAME}"
echo "  Attach: tmux attach -t ${SESSION_NAME}"
echo "  Log: ${LOG_FILE}"
