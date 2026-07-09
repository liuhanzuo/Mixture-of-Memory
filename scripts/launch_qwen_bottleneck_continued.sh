#!/usr/bin/env bash
# Continued-pretrain a REAL Qwen3-8B with a mid-layer semantic funnel (8-GPU DDP).
#
# Two arms:
#   funnel arm (default): --bottleneck_dim 512 --bottleneck_layer 12
#   baseline arm        : ARM=baseline  (BOTTLENECK_DIM=0 -> stock-continued, no funnel)
#
# Usage (8-GPU single node, L20A 183GB or H20 97GB):
#   bash scripts/launch_qwen_bottleneck_continued.sh
#   ARM=baseline PORT=29621 OUT=outputs/qwenbott_baseline bash scripts/launch_qwen_bottleneck_continued.sh
#   BOTTLENECK_LAYER=12 BOTTLENECK_DIM=512 SEQ_LEN=2048 BATCH_SIZE=4 GRAD_ACCUM=4 \
#     bash scripts/launch_qwen_bottleneck_continued.sh
set -euo pipefail

export PATH=/opt/conda/bin:$PATH
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
cd "$PROJECT_ROOT"

export WANDB_MODE=offline
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false
# single-node: disable IB, use loopback rdzv.
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b}"
ARM="${ARM:-funnel}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-29620}"
DATA_PATH="${DATA_PATH:-data/pg19_train.jsonl}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-2000}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP="${WARMUP:-100}"
BOTTLENECK_LAYER="${BOTTLENECK_LAYER:-12}"
UNFREEZE_FROM="${UNFREEZE_FROM:--1}"   # -1 => defaults to BOTTLENECK_LAYER; 0 => full continued
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [ "$ARM" = "baseline" ]; then
  BD=0
else
  BD="${BOTTLENECK_DIM:-512}"
fi

NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OUT="${OUT:-outputs/qwenbott_${ARM}_L${BOTTLENECK_LAYER}_d${BD}}"
mkdir -p logs "$OUT"

echo "[launch] MODEL=$MODEL_PATH ARM=$ARM GPUS=$GPUS ngpu=$NGPU bd=$BD layer=$BOTTLENECK_LAYER unfreeze=$UNFREEZE_FROM out=$OUT port=$PORT"
echo "[launch] seq_len=$SEQ_LEN bs=$BATCH_SIZE grad_accum=$GRAD_ACCUM eff_bs=$((BATCH_SIZE*GRAD_ACCUM*NGPU)) lr=$LR max_steps=$MAX_STEPS"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NGPU" --nnodes=1 \
  --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  scripts/train_qwen_bottleneck_continued.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT" \
  --bottleneck_layer "$BOTTLENECK_LAYER" \
  --bottleneck_dim "$BD" \
  --unfreeze_from "$UNFREEZE_FROM" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" --min_lr "$MIN_LR" --warmup_steps "$WARMUP" \
  --max_steps "$MAX_STEPS" \
  --save_every "$SAVE_EVERY" --log_every "$LOG_EVERY" \
  --num_workers "$NUM_WORKERS"
