#!/usr/bin/env bash
# Phase-3 S4b (retrieval/gating axis) — per-node runner. Group-A 2-node.
#   Invoked on BOTH nodes with $1=node_rank, $2=log, $3=max_steps (default 3000).
#   Single-axis change vs reproduced Landmark anchor: ONLY the block-gating
#   FUNCTION changes — parameter-free grouped-softmax -> learned soft per-block
#   scalar gate (config.learned_block_gate=True; gate=exp(MLP(landmark_hidden)),
#   final layer zero-init => exp(0)=1 == bit-identical to grouped-softmax at step0).
#   Everything else identical to the anchor / S2 recipe:
#     LLaMA-1-7B base, RedPajama-1T-Sample mirror (liang2kl backup), mem_freq=63,
#     grouped-softmax all layers, lr2e-5 cosine+3% warmup, wd0.1, bf16, FSDP,
#     ctx512 single window, all-token LM loss.
#   eff-batch = 128 = per_device2 x accum4 x 16 ranks (2 nodes x 8 GPU).
#   max_steps default 3000, save_steps 1000 (matches S0/S2 gating budget).
# Verified NCCL recipe (Group-A diskA): static rdzv + bond1 + IB/RoCE GID3 (11x TCP).
set -euo pipefail

NODE_RANK="${1:?usage: run_landmark_S4b_node.sh <node_rank> <log> [max_steps]}"
LOG="${2:?usage: run_landmark_S4b_node.sh <node_rank> <log> [max_steps]}"
MAX_STEPS="${3:-3000}"

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
case "$LOG" in
  /*) : ;;
  *) LOG="$PROJECT_ROOT/$LOG" ;;
esac
mkdir -p "$(dirname "$LOG")"
REPO="$PROJECT_ROOT/external/landmark-attention/llama"
PY="$PROJECT_ROOT/external/landmark_venv/bin/torchrun"
BASE="$PROJECT_ROOT/external/landmark_ckpts/llama1_7b_base"
OUT="${OUT:-$PROJECT_ROOT/external/landmark_ckpts/landmark_S4b_learnedgate}"
CACHE="$PROJECT_ROOT/external/landmark/hf-cache"
MASTER_IP="29.162.227.178"
PORT="${PORT:-29553}"

# --- proxy: RedPajama mirror may need fetch on first call (cached after) ---
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export HF_HOME="${HF_HOME:-$CACHE/hf_home}"

# --- Group-A NCCL recipe (verified S2): IB/RoCEv2 GID3 ~11x TCP ---
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export WANDB_MODE=offline
# S4B_GATE_DIAG passthrough (smoke uses =1; 3k run leaves unset/0)
export S4B_GATE_DIAG="${S4B_GATE_DIAG:-0}"

cd "$REPO"
setsid nohup "$PY" \
  --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node 8 \
  --master_addr "$MASTER_IP" --master_port "$PORT" \
  train.py \
  --model_name_or_path "$BASE" \
  --bf16 True \
  --output_dir "$OUT" \
  --cache_dir "$CACHE" \
  --learned_block_gate True \
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
  --max_steps "$MAX_STEPS" > "$LOG" 2>&1 &
echo "NODE${NODE_RANK}_PID=$!"
