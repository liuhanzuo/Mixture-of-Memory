#!/usr/bin/env bash
# Launch one arm of the minimal-architecture probe #2 (from-scratch/continue-train
# "front-j inherited + k fresh NTP layers"). See scripts/train_minimal_arch_probe2.py.
#
# Three arms (via ARM):
#   armA        : freeze inherited front layers, train fresh + norm + head only
#   armB        : train all 14 layers (healing)                       [default]
#   scratch14   : ignore ckpt, random-init 14 layers, train all (control 2)
#
# Usage:
#   ARM=armB      GPUS=0,1,2,3 PORT=29620 bash scripts/launch_minimal_arch_probe2.sh
#   ARM=armA      GPUS=4,5,6,7 PORT=29621 bash scripts/launch_minimal_arch_probe2.sh
#   ARM=scratch14 GPUS=0,1,2,3 PORT=29622 bash scripts/launch_minimal_arch_probe2.sh
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
MODEL_SIZE="${MODEL_SIZE:-1b}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-29620}"
DATA_PATH="${DATA_PATH:-data/slimpajama_chunks_4096_llama3.npy}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-2000}"
LR="${LR:-3e-4}"
KEEP_FRONT="${KEEP_FRONT:-12}"
N_FRESH="${N_FRESH:-2}"
INIT_CKPT="${INIT_CKPT:-outputs/sembott_1b_base_16k/final.pt}"

ARM_FLAGS=""
case "$ARM" in
  armA)
    ARM_FLAGS="--freeze_front"
    ;;
  armB)
    ARM_FLAGS=""
    ;;
  scratch14)
    ARM_FLAGS="--from_scratch"
    ;;
  *)
    echo "[launch] unknown ARM=$ARM (expected armA|armB|scratch14)" >&2
    exit 1
    ;;
esac

NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OUT="${OUT:-outputs/minarch_${MODEL_SIZE}_${ARM}_f${KEEP_FRONT}k${N_FRESH}}"
mkdir -p logs "$OUT"

echo "[launch] MODEL_SIZE=$MODEL_SIZE ARM=$ARM keep_front=$KEEP_FRONT n_fresh=$N_FRESH GPUS=$GPUS ngpu=$NGPU out=$OUT port=$PORT init_ckpt=$INIT_CKPT"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NGPU" --nnodes=1 \
  --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  scripts/train_minimal_arch_probe2.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT" \
  --init_ckpt "$INIT_CKPT" \
  --model_size "$MODEL_SIZE" \
  --keep_front_layers "$KEEP_FRONT" \
  --n_fresh_layers "$N_FRESH" \
  $ARM_FLAGS \
  --max_steps "$MAX_STEPS" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUM" \
  --lr "$LR" --min_lr 3e-5 --warmup_steps 100 \
  --save_every 500 --log_every 20
