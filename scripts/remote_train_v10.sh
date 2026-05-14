#!/bin/bash
# RMT++ v10 launcher for current shared wzc1 project tree.
# Usage: bash scripts/remote_train_v10.sh --config <l0|l0l1|l0l2|l0l1l2> [--output_dir <dir>] [--master_port <port>]

set -euo pipefail

export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
TORCHRUN_CMD="${TORCHRUN_CMD:-${PYTHON_BIN} -m torch.distributed.run}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/legacy/scripts/train_rmt_v10.py}"
DATA="${DATA:-${PROJECT_ROOT}/data/rmt_train_mixed.jsonl}"
BASE_MODEL="${BASE_MODEL:-/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama2-7b}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29510}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "$LOG_DIR"

CONFIG=""
EXTRA_OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --output_dir) EXTRA_OUTPUT="$2"; shift 2 ;;
    --master_port) MASTER_PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Usage: $0 --config <l0|l0l1|l0l2|l0l1l2>" >&2
  exit 1
fi

case "$CONFIG" in
  l0)      OUTPUT_DIR="${EXTRA_OUTPUT:-${PROJECT_ROOT}/outputs/rmt_v10_l0}" ;;
  l0l1)    OUTPUT_DIR="${EXTRA_OUTPUT:-${PROJECT_ROOT}/outputs/rmt_v10_l0l1}" ;;
  l0l2)    OUTPUT_DIR="${EXTRA_OUTPUT:-${PROJECT_ROOT}/outputs/rmt_v10_l0l2}" ;;
  l0l1l2)  OUTPUT_DIR="${EXTRA_OUTPUT:-${PROJECT_ROOT}/outputs/rmt_v10_l0l1l2}" ;;
  *)       echo "Unknown config: $CONFIG" >&2; exit 1 ;;
esac

L1L2_FLAGS=""
case "$CONFIG" in
  l0)      ;;
  l0l1)    L1L2_FLAGS="--use_l1" ;;
  l0l2)    L1L2_FLAGS="--use_l2" ;;
  l0l1l2)  L1L2_FLAGS="--use_l1 --use_l2" ;;
esac

LOG_FILE="$LOG_DIR/rmt_v10_${CONFIG}_$(date +%Y%m%d_%H%M).log"

cd "$PROJECT_ROOT"

CMD="${TORCHRUN_CMD} \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  ${TRAIN_SCRIPT} \
  --data ${DATA} \
  --output_dir ${OUTPUT_DIR} \
  --base_model ${BASE_MODEL} \
  --num_mem_tokens 16 \
  --segment_length 1024 \
  --max_segments 4 \
  --vary_n_segments \
  --bptt_depth 2 \
  --recon_loss_coef 0.1 \
  --use_importance_routing \
  --full_finetune \
  --num_epochs 5 \
  --lr 1e-5 \
  --rmt_lr 1e-4 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --warmup_steps 30 \
  --log_every 10 \
  --save_every 100 \
  --eval_every 100 \
  --seed 42 \
  ${L1L2_FLAGS} \
  --ddp"

{
  echo "========================================"
  echo "RMT v10 Training"
  echo "Project root: $PROJECT_ROOT"
  echo "Config:       $CONFIG"
  echo "Output:       $OUTPUT_DIR"
  echo "Train script: $TRAIN_SCRIPT"
  echo "Python:       $PYTHON_BIN"
  echo "Torchrun cmd: $TORCHRUN_CMD"
  echo "GPUs:         $NUM_GPUS"
  echo "Master port:  $MASTER_PORT"
  echo "Extra flags:  $L1L2_FLAGS"
  echo "Log:          $LOG_FILE"
  echo "========================================"
} | tee "$LOG_FILE"

bash -lc "$CMD" 2>&1 | tee -a "$LOG_FILE"
