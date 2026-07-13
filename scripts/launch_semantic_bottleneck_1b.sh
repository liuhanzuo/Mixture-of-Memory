#!/usr/bin/env bash
# Launch one arm of the semantic-bottleneck feasibility experiment on 4 GPUs.
# Supports the 1b/3b/7b scale-up ladder via MODEL_SIZE.
#
# Usage:
#   ARM=baseline    GPUS=0,1,2,3 PORT=29610 bash scripts/launch_semantic_bottleneck_1b.sh
#   ARM=bottleneck  GPUS=4,5,6,7 PORT=29611 bash scripts/launch_semantic_bottleneck_1b.sh
#   MODEL_SIZE=3b ARM=baseline GPUS=0,1,2,3,4,5,6,7 PORT=29612 BATCH_SIZE=8 bash scripts/launch_semantic_bottleneck_1b.sh
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

ARM="${ARM:-baseline}"
MODEL_SIZE="${MODEL_SIZE:-1b}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-29610}"
DATA_PATH="${DATA_PATH:-data/slimpajama_chunks_4096_llama3.npy}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-2000}"
LR="${LR:-3e-4}"
BOTTLENECK_LAYER="${BOTTLENECK_LAYER:-6}"

if [ "$ARM" = "baseline" ]; then
  BD=0
else
  BD="${BOTTLENECK_DIM:-512}"
fi

NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OUT="${OUT:-outputs/sembott_${MODEL_SIZE}_${ARM}}"
mkdir -p logs "$OUT"

echo "[launch] MODEL_SIZE=$MODEL_SIZE ARM=$ARM GPUS=$GPUS ngpu=$NGPU bd=$BD layer=$BOTTLENECK_LAYER out=$OUT port=$PORT"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NGPU" --nnodes=1 \
  --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  scripts/train_semantic_bottleneck_1b.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT" \
  --model_size "$MODEL_SIZE" \
  --bottleneck_layer "$BOTTLENECK_LAYER" \
  --bottleneck_dim "$BD" \
  --max_steps "$MAX_STEPS" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUM" \
  --lr "$LR" --min_lr 3e-5 --warmup_steps 100 \
  --save_every 500 --log_every 20 \
  ${RESUME_FROM:+--resume_from "$RESUME_FROM"}
