#!/bin/bash
# Phase-3 S2 (data axis) — Landmark mechanism on LLaMA-1-7B, corpus = dolmino wiki+pes2o
# (raw text re-tokenized with the SAME LLaMA-1 tokenizer). SINGLE AXIS vs the anchor.
#
# Everything pinned to the reproduced anchor recipe EXCEPT the corpus:
#   base LLaMA-1-7B, LLaMA-1 SP tokenizer, ctx512, 512-block packing, all-token LM loss,
#   mem_freq=63, grouped-softmax untouched, lr2e-5 cosine + 3% warmup, wd0.1,
#   eff-batch=128 (per_device2 x accum4 x 16 ranks).
#   Diagnostic scope: max_steps 3000, save_steps 1000.
#
# 2-node DDP Group-A: master = 本机 (29.162.227.178), worker = 28.59.80.196 (diskA shared FS).
# Run THIS script on master AND worker (set NODE_RANK accordingly).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/external/landmark_venv/bin/python}"
cd "$PROJECT_ROOT"

MASTER_ADDR="${MASTER_ADDR:-29.162.227.178}"
MASTER_PORT="${MASTER_PORT:-29517}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
NPROC="${NPROC:-8}"

BASE_CKPT="$PROJECT_ROOT/external/landmark_ckpts/llama1_7b_base"
OUT_DIR="$PROJECT_ROOT/external/landmark_ckpts/s2_dolmino_3k"
export S2_DATA_FILE="$PROJECT_ROOT/MemLong/data/processed/landmark_s2_dolmino_raw/train.jsonl"
export WANDB_MODE=offline
export HF_DATASETS_CACHE="$PROJECT_ROOT/external/landmark-attention/llama/.hf_cache_s2"
# cross-node NCCL hardening (per cluster notes for diskA inter-node)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"

mkdir -p "$OUT_DIR"

cd "$PROJECT_ROOT/external/landmark-attention/llama"

torchrun \
  --nnodes "$NNODES" --nproc_per_node "$NPROC" --node_rank "$NODE_RANK" \
  --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  train_s2.py \
  --model_name_or_path "$BASE_CKPT" \
  --bf16 True \
  --output_dir "$OUT_DIR" \
  --cache_dir "$HF_DATASETS_CACHE" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --model_max_length 512 \
  --mem_freq 63 \
  --max_steps 3000
