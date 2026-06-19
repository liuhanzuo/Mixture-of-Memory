#!/usr/bin/env bash
# Phase-3 S5 (single-layer readout axis) — per-node runner. Group-A 2-node.
#   Invoked on BOTH nodes with $1=node_rank, $2=log, $3=max_steps (default 2000).
#   Runs FROM the ISOLATED S5 tree external/landmark_s5_tree/llama/ (pristine
#   anchor 99631a8 + single_layer_mem patch only; physically separate from the
#   S4b live tree). Single axis = LM_SINGLE_LAYER=16 -> only L16 runs landmark
#   grouped-softmax, other 31 layers run plain causal softmax over the same KV.
#   Everything else = anchor: LLaMA-1-7B, RedPajama liang2kl mirror, mem_freq63,
#   grouped-softmax readout, lr2e-5 cosine+3% warmup, wd0.1, bf16, FSDP, ctx512,
#   single 512 window, grad-ckpt OFF (the landmark grad-ckpt path is buggy — passes
#   use_cache as a tensor at llama_mem.py:460 → "Boolean value of Tensor ambiguous";
#   faithful S2/S4b anchor also runs grad-ckpt OFF; 7B ctx512 single-window FSDP fits).
#   eff-batch 128 = per_device2 x grad_accum4 x 16 ranks (2 nodes x 8 GPU).
#   ★max_steps default 2000 save_steps 500 (judge step1000/2000, avoid step3000
#   overtraining collapse per project rule).
# Group-A NCCL: static rdzv + bond1 + IB/RoCE GID3 + P2P_DISABLE=1.
set -euo pipefail

NODE_RANK="${1:?usage: run_landmark_S5_node.sh <node_rank> <log> [max_steps]}"
LOG="${2:?usage: run_landmark_S5_node.sh <node_rank> <log> [max_steps]}"
MAX_STEPS="${3:-2000}"

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
case "$LOG" in
  /*) : ;;
  *) LOG="$PROJECT_ROOT/$LOG" ;;
esac
mkdir -p "$(dirname "$LOG")"
S5TREE="$PROJECT_ROOT/external/landmark_s5_tree/llama"
PY="$PROJECT_ROOT/external/landmark_venv/bin/torchrun"
BASE="$PROJECT_ROOT/external/landmark_ckpts/llama1_7b_base"
OUT="${OUT:-$PROJECT_ROOT/external/landmark_ckpts/landmark_S5_L16_singlelayer}"
CACHE="$PROJECT_ROOT/external/landmark/hf-cache"          # RedPajama mirror cache (S4b populated)
MASTER_IP="29.162.227.178"
PORT="${PORT:-29581}"
SAVE_STEPS="${SAVE_STEPS:-500}"

# --- S5 axis ---
export LM_SINGLE_LAYER="${LM_SINGLE_LAYER:-16}"

# --- proxy for RedPajama dataset metadata (data already cached) ---
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export HF_HOME="${HF_HOME:-$CACHE/hf_home}"
export DATASET_MAP_NPROC="${DATASET_MAP_NPROC:-8}"
export TOKENIZERS_PARALLELISM=false

# --- Group-A NCCL recipe (verified S2/S4b): IB/RoCEv2 GID3 ---
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export WANDB_MODE=offline

mkdir -p "$OUT"
cd "$S5TREE"
echo "[S5] NODE_RANK=$NODE_RANK MASTER=$MASTER_IP:$PORT LM_SINGLE_LAYER=$LM_SINGLE_LAYER MAX_STEPS=$MAX_STEPS SAVE=$SAVE_STEPS"
echo "[S5] tree=$S5TREE BASE=$BASE OUT=$OUT"
setsid nohup "$PY" \
  --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node 8 \
  --master_addr "$MASTER_IP" --master_port "$PORT" \
  train.py \
  --model_name_or_path "$BASE" \
  --bf16 True \
  --output_dir "$OUT" \
  --cache_dir "$CACHE" \
  --model_max_length 512 \
  --mem_freq 63 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  --tf32 True \
  --report_to none \
  --max_steps "$MAX_STEPS" > "$LOG" 2>&1 &
echo "NODE${NODE_RANK}_PID=$!"
