#!/usr/bin/env bash
# Phase-3 S2 (data axis) — per-node runner. Invoked on BOTH nodes with $1=node_rank.
#   Single-axis change vs reproduced Landmark anchor: training corpus only
#   (RedPajama-1T-Sample -> dolmino wiki+pes2o, raw text, LLaMA-1 tokenizer).
#   Everything else identical: LLaMA-1-7B base, mem_freq=63, grouped-softmax,
#   lr2e-5 cosine+3% warmup, wd0.1, bf16, FSDP, ctx512, all-token LM loss.
#   eff-batch = 128 = per_device2 x accum4 x 16 ranks (2 nodes x 8 GPU).
#   Diagnostic budget: max_steps=3000, save_steps=1000.
# Verified NCCL recipe (Group-A diskA): static rdzv + bond1 + IB disabled.
set -euo pipefail

NODE_RANK="${1:?usage: run_landmark_S2_node.sh <node_rank> <log>}"
LOG="${2:?usage: run_landmark_S2_node.sh <node_rank> <log>}"

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
# normalize LOG to an absolute path (we cd into $REPO below, so relative paths break)
case "$LOG" in
  /*) : ;;
  *) LOG="$PROJECT_ROOT/$LOG" ;;
esac
mkdir -p "$(dirname "$LOG")"
REPO="$PROJECT_ROOT/external/landmark-attention/llama"
PY="$PROJECT_ROOT/external/landmark_venv/bin/torchrun"
BASE="$PROJECT_ROOT/external/landmark_ckpts/llama1_7b_base"
DATA="$PROJECT_ROOT/MemLong/data/processed/landmark_s2_dolmino_raw/train.jsonl"
OUT="$PROJECT_ROOT/external/landmark_ckpts/landmark_s2_dolmino"
MASTER_IP="29.162.227.178"
PORT="29551"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export WANDB_MODE=offline
export S2_DATA_FILE="$DATA"

cd "$REPO"
setsid nohup "$PY" \
  --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node 8 \
  --master_addr "$MASTER_IP" --master_port "$PORT" \
  train_s2.py \
  --model_name_or_path "$BASE" \
  --bf16 True \
  --output_dir "$OUT" \
  --cache_dir "$PROJECT_ROOT/external/landmark/hf-cache" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 4 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  --tf32 True \
  --report_to none \
  --mem_freq 63 \
  --model_max_length 512 \
  --max_steps 3000 > "$LOG" 2>&1 &
echo "NODE${NODE_RANK}_PID=$!"
