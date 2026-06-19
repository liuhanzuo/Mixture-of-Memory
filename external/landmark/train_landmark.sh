#!/usr/bin/env bash
# Phase 3 S5 (and general Landmark from-base retrain) launcher.
# Faithfully mirrors the OFFICIAL recipe (external/landmark-attention/README.md):
#   torchrun --nproc_per_node=8 train.py, bf16, per_device 2 x grad_accum 8 x 8gpu = eff-batch 128,
#   lr 2e-5 cosine, wd 0.1, warmup_ratio 0.03, max_steps 15000, FSDP full_shard auto_wrap,
#   model_max_length(ctx) 512, mem_freq 63 (train.py default), RedPajama-1T-Sample, full-FT.
# The ONLY axis S5 changes is INSIDE llama_mem.py (which layers run landmark attn) — NOT here.
#
# Env knobs:
#   PROJECT_ROOT  : repo root on this node (diskB: /apdcephfs_zwfy6/share_304376610/.../Mixture-of-Memory)
#   PY            : path to external/landmark_venv/bin/python (tf4.28.1 + torch2.1.0cu121)
#   BASE          : LLaMA-1-7B HF base ckpt dir
#   OUT           : output dir for the retrained ckpt
#   CACHE         : hf-cache dir (RedPajama download + model cache)
#   MEM_FREQ      : landmark frequency (default 63 = official from-base recipe)
#   MAX_STEPS     : default 15000 (~1 epoch RedPajama-1T-Sample)
#   LM_SINGLE_LAYER : (S5) if set, restrict landmark attn to this layer idx (read by llama_mem.py)
#   NPROC         : GPUs (default 8)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$(cd "$HERE/.." && pwd)"
REPO="$EXT/landmark-attention/llama"

PY="${PY:-$EXT/landmark_venv/bin/python}"
BASE="${BASE:-$EXT/landmark_ckpts/llama1_7b_base}"
OUT="${OUT:-$EXT/landmark_ckpts/landmark_S5_singlelayer}"
CACHE="${CACHE:-$HERE/hf-cache}"
MEM_FREQ="${MEM_FREQ:-63}"
MAX_STEPS="${MAX_STEPS:-15000}"
NPROC="${NPROC:-8}"

# diskB nodes have no internet without proxy; RedPajama download needs woa proxy.
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export HF_HOME="${HF_HOME:-$CACHE/hf_home}"
export WANDB_MODE=offline
# S5 axis: passed through env to llama_mem.py (no effect if unset = vanilla all-layer recipe).
export LM_SINGLE_LAYER="${LM_SINGLE_LAYER:-}"

mkdir -p "$OUT" "$CACHE"
cd "$REPO"

echo "[train_landmark] PY=$PY"
echo "[train_landmark] BASE=$BASE"
echo "[train_landmark] OUT=$OUT  MEM_FREQ=$MEM_FREQ  MAX_STEPS=$MAX_STEPS  NPROC=$NPROC"
echo "[train_landmark] LM_SINGLE_LAYER=${LM_SINGLE_LAYER:-<unset=all-layer>}"

exec torchrun --nproc_per_node="$NPROC" train.py \
    --model_name_or_path "$BASE" \
    --bf16 True \
    --output_dir "$OUT" \
    --cache_dir "$CACHE" \
    --model_max_length 512 \
    --mem_freq "$MEM_FREQ" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --max_steps "$MAX_STEPS"
