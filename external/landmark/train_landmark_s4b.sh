#!/usr/bin/env bash
# Phase 3 S4b: learned soft block-gate (structural grouped-softmax -> learned).
# Single axis: ONLY the gating function changes (config.learned_block_gate=True);
# everything else mirrors the faithful Landmark from-base anchor recipe.
# 2-node Group-B (.76 master + .249 worker), 16 ranks, IB recipe, static rdzv.
#
# Per-NODE env knobs:
#   NODE_RANK   : 0 (master .76) or 1 (worker .249)   [REQUIRED]
#   MASTER_ADDR : 28.49.57.76 (default)
#   PROJECT_ROOT: diskB repo root (default below)
# Recipe: per_device 2 x grad_accum 4 x 16 ranks = eff-batch 128, lr 2e-5 cosine,
#   wd 0.1, warmup 0.03, ctx512, mem_freq63, RedPajama mirror, NO grad-ckpt
#   (faithful anchor: single-512-window 7B fits ~16GB/GPU under FSDP; and the
#   landmark grad-ckpt path is buggy — passes use_cache as a tensor at :460),
#   max_steps 3000, save_steps 1000, full-FT, FSDP full_shard.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
EXT="$PROJECT_ROOT/external"
REPO="$EXT/landmark-attention/llama"
PY="${PY:-$EXT/landmark_venv/bin/python}"
BASE="${BASE:-$EXT/landmark_ckpts/llama1_7b_base}"
OUT="${OUT:-$EXT/landmark_ckpts/landmark_S4b_learnedgate}"
CACHE="${CACHE:-$EXT/landmark/hf-cache}"
MEM_FREQ="${MEM_FREQ:-63}"
MAX_STEPS="${MAX_STEPS:-3000}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on .76 master, 1 on .249 worker}"
MASTER_ADDR="${MASTER_ADDR:-28.49.57.76}"
MASTER_PORT="${MASTER_PORT:-29560}"

# --- proxy for RedPajama mirror download (diskB no direct internet) ---
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export HF_HOME="${HF_HOME:-$CACHE/hf_home}"
export WANDB_MODE=offline

# --- Group-B NCCL recipe (verified): IB/RoCE 17 GB/s + intra-node P2P off ---
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DMABUF_ENABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_DISABLE=1
export GLOO_SOCKET_IFNAME=bond1

mkdir -p "$OUT" "$CACHE"
cd "$REPO"

echo "[S4b] NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT NNODES=$NNODES NPROC=$NPROC"
echo "[S4b] PY=$PY BASE=$BASE OUT=$OUT MAX_STEPS=$MAX_STEPS learned_block_gate=True"

# NOTE: launch via `python -m torch.distributed.run` NOT venv torchrun
# (venv torchrun shebang hardcodes diskA python path).
exec "$PY" -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train.py \
    --model_name_or_path "$BASE" \
    --bf16 True \
    --output_dir "$OUT" \
    --cache_dir "$CACHE" \
    --model_max_length 512 \
    --mem_freq "$MEM_FREQ" \
    --learned_block_gate True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
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
    --max_steps "$MAX_STEPS"
