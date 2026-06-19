#!/usr/bin/env bash
# Phase-3 S2 (data axis) 2-node DDP launch on Group-A.
#   Single-axis change vs the reproduced Landmark anchor: training corpus only
#   (RedPajama-1T-Sample -> dolmino wiki+pes2o, raw text, LLaMA-1 tokenizer).
#   Everything else identical: LLaMA-1-7B base, mem_freq=63, grouped-softmax,
#   lr2e-5 cosine+3% warmup, wd0.1, bf16, FSDP, ctx512, all-token LM loss.
#   eff-batch = 128 = per_device2 x accum4 x 16 ranks (2 nodes x 8 GPU).
#   Diagnostic budget: max_steps=3000, save_steps=1000.
#
# Verified NCCL recipe for Group-A diskA nodes (see landmark-repro findings):
#   static rendezvous (--master_addr/--master_port, NOT c10d),
#   NCCL_SOCKET_IFNAME=bond1, NCCL_IB_DISABLE=1, worker launched after master.
#
# Usage (run from PROJECT_ROOT on 本机 = master):
#   bash scripts/launch_landmark_S2.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
REPO="$PROJECT_ROOT/external/landmark-attention/llama"
PY="$PROJECT_ROOT/external/landmark_venv/bin"
BASE="$PROJECT_ROOT/external/landmark_ckpts/llama1_7b_base"
DATA="$PROJECT_ROOT/MemLong/data/processed/landmark_s2_dolmino_raw/train.jsonl"
OUT="$PROJECT_ROOT/external/landmark_ckpts/landmark_s2_dolmino"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs"

MASTER_IP="29.162.227.178"
WORKER_IP="28.59.80.196"
PORT="${PORT:-29551}"
PASS="$PROJECT_ROOT/configs/password_diskA.txt"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true

COMMON_ARGS="train_s2.py \
  --model_name_or_path $BASE \
  --bf16 True \
  --output_dir $OUT \
  --cache_dir $PROJECT_ROOT/external/landmark/hf-cache \
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
  --fsdp 'full_shard auto_wrap' \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  --tf32 True \
  --mem_freq 63 \
  --model_max_length 512 \
  --max_steps 3000"

WORKER_LOG="$LOG_DIR/landmark_S2_dolmino_worker_${TS}.log"
MASTER_LOG="$LOG_DIR/landmark_S2_dolmino_master_${TS}.log"

echo "S2 launch: master=$MASTER_IP worker=$WORKER_IP port=$PORT"
echo "data=$DATA"
echo "out=$OUT"
echo "master_log=$MASTER_LOG"
echo "worker_log=$WORKER_LOG"

# 1) Master (本机) in background
cd "$REPO"
export S2_DATA_FILE="$DATA"
setsid nohup "$PY/torchrun" \
  --nnodes 2 --node_rank 0 --nproc_per_node 8 \
  --master_addr "$MASTER_IP" --master_port "$PORT" \
  $COMMON_ARGS > "$MASTER_LOG" 2>&1 &
MASTER_PID=$!
echo "MASTER_PID=$MASTER_PID"

# 2) Give the master store time to bind before the worker connects
sleep 12

# 3) Worker (.196) via SSH, detached
sshpass -f "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -o PreferredAuthentications=password root@"$WORKER_IP" \
  "cd $REPO && export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=bond1 NCCL_IB_DISABLE=1 WANDB_MODE=offline WANDB_DISABLED=true S2_DATA_FILE='$DATA' && setsid nohup $PY/torchrun --nnodes 2 --node_rank 1 --nproc_per_node 8 --master_addr $MASTER_IP --master_port $PORT $COMMON_ARGS > $WORKER_LOG 2>&1 & echo WORKER_PID=\$!"

echo "launched. master_pid=$MASTER_PID"
echo "$MASTER_PID" > "$LOG_DIR/landmark_S2_master.pid"
