#!/usr/bin/env bash
# Launch one arm of the Qwen3-8B minimal-architecture probe #2
# ("front-j pretrained layers + k fresh NTP layers", continue-train).
# See scripts/train_qwen3_arch_probe2.py.
#
# Three arms (via ARM):
#   armA     : freeze inherited front layers, train fresh + norm + head only
#   armB     : train ALL layers (healing, differential LR)            [default]
#   scratch  : ignore base weights, random-init all layers, train all (control 2)
# Control 0 (full 36-layer Qwen3-8B) needs no training -> eval via eval_qwen_ppl.py.
#
# Usage:
#   ARM=armB    KEEP_FRONT=12 N_FRESH=2 GPUS=0,1,2,3,4,5,6,7 PORT=29650 \
#     bash scripts/launch_qwen3_arch_probe2.sh
#   ARM=armA    KEEP_FRONT=12 N_FRESH=4 GPUS=0,1,2,3 PORT=29651 \
#     bash scripts/launch_qwen3_arch_probe2.sh
#   ARM=scratch KEEP_FRONT=12 N_FRESH=2 GPUS=4,5,6,7 PORT=29652 \
#     bash scripts/launch_qwen3_arch_probe2.sh
set -euo pipefail

export PATH=/opt/conda/bin:$PATH
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
cd "$PROJECT_ROOT"

export WANDB_MODE=offline
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false
# L20A single-node: disable IB, use loopback rdzv.
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

ARM="${ARM:-armB}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-29650}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b}"
DATA_PATH="${DATA_PATH:-data/slimpajama_chunks_2048_qwen3.npy}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-2000}"
LR="${LR:-1e-4}"
LR_INHERITED="${LR_INHERITED:-2e-5}"
KEEP_FRONT="${KEEP_FRONT:-12}"
N_FRESH="${N_FRESH:-2}"

ARM_FLAGS=""
case "$ARM" in
  armA)    ARM_FLAGS="--freeze_front" ;;
  armB)    ARM_FLAGS="" ;;
  scratch) ARM_FLAGS="--from_scratch" ;;
  *)
    echo "[launch] unknown ARM=$ARM (expected armA|armB|scratch)" >&2
    exit 1
    ;;
esac

NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OUT="${OUT:-outputs/qwen3_minarch_${ARM}_f${KEEP_FRONT}k${N_FRESH}}"
mkdir -p logs "$OUT"

echo "[launch] qwen3 ARM=$ARM keep_front=$KEEP_FRONT n_fresh=$N_FRESH GPUS=$GPUS ngpu=$NGPU out=$OUT port=$PORT model=$MODEL_PATH"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NGPU" --nnodes=1 \
  --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  scripts/train_qwen3_arch_probe2.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT" \
  --model_path "$MODEL_PATH" \
  --keep_front_layers "$KEEP_FRONT" \
  --n_fresh_layers "$N_FRESH" \
  $ARM_FLAGS \
  --max_steps "$MAX_STEPS" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUM" \
  --lr "$LR" --lr_inherited "$LR_INHERITED" \
  --warmup_steps 150 \
  --save_every 500 --log_every 20
