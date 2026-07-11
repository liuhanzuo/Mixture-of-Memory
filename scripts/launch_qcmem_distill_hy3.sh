#!/usr/bin/env bash
# QCMem self-distillation on Hy3 (hy_v3, 80-layer MoE, 597 GB) — 2026-07-12.
# Teacher = QCMem read at j=0 (RAG upper bound, adapters OFF).
# Student = QCMem read at j=RESUME_J (default 32 = Hy3 split-j) + LoRA r32/a64 on
# layers[j:]. Loss = bidirectional top-64 KL on the QUERY-tail tokens over pure
# PG19 natural text (NO babilong / NO needles / NO synthetic — red line).
#
# Goal: push the zero-shot j=32 LM tax (1.25-1.5x, from the j-sweep) toward 1.0,
# mirroring the Qwen 8B self-distill that lifted every qa cell.
#
# CRITICAL: Hy3 (597 GB) does NOT fit on one GPU -> ONE device_map="auto" sharded
# instance, SINGLE process (NO torchrun / NO DDP). Every forward already pipelines
# across all 8 L20A, so a single process saturates the node.
#
# === USAGE (local B200 node, 8x L20A 183GB, wzc1) ===========================
#   PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory \
#   setsid nohup bash scripts/launch_qcmem_distill_hy3.sh \
#     >logs/hy3_distill_j32.out 2>&1 &
# Override defaults via env: RESUME_J=32 N_CTX=3 TOTAL_STEPS=4000 LR=8e-5 ...
# ============================================================================
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# .venv_hy3 = torch 2.10 (L20A sm_100) + transformers 5.13.1 (knows hy_v3).
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_hy3/bin/python}"

MODEL="${MODEL:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Hy3}"
RESUME_J="${RESUME_J:-32}"
TOP_PREPAY_B="${TOP_PREPAY_B:-0}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
N_CTX="${N_CTX:-3}"                 # (3+1)*512 = 2048-tok training window (Qwen recipe)
TEACHER_TOPK="${TEACHER_TOPK:-64}"
DISTILL_LAMBDA="${DISTILL_LAMBDA:-0.6}"
CE_WEIGHT="${CE_WEIGHT:-0.0}"
TOTAL_STEPS="${TOTAL_STEPS:-4000}"
LR="${LR:-8e-5}"
WARMUP="${WARMUP:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEVS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GC_FLAG="${GRADIENT_CHECKPOINTING:-}"   # set to "1" to add --gradient_checkpointing
RUN="${RUN:-qcmem_distill_hy3_j${RESUME_J}_r${LORA_RANK}}"
OUTPUT_DIR="outputs/$RUN"
mkdir -p logs "$OUTPUT_DIR"

GC_ARG=""
[ -n "$GC_FLAG" ] && GC_ARG="--gradient_checkpointing"

echo "[launch] RUN=$RUN model=$MODEL j=$RESUME_J b=$TOP_PREPAY_B r${LORA_RANK}/a${LORA_ALPHA} n_ctx=$N_CTX steps=$TOTAL_STEPS devs=$DEVS gc='$GC_ARG'"

setsid bash -c "CUDA_VISIBLE_DEVICES=$DEVS \
  $PYBIN scripts/train_qcmem_distill_hy3.py \
  --model_path $MODEL \
  --resume_j $RESUME_J --top_prepay_b $TOP_PREPAY_B \
  --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \
  --chunk_size $CHUNK_SIZE --n_ctx $N_CTX \
  --teacher_topk $TEACHER_TOPK --distill_lambda $DISTILL_LAMBDA --ce_weight $CE_WEIGHT \
  --total_steps $TOTAL_STEPS --lr $LR --warmup_steps $WARMUP --grad_accum $GRAD_ACCUM \
  $GC_ARG \
  --output_dir $OUTPUT_DIR --save_interval 500 --log_interval 10 \
  --dtype bfloat16 --attn_impl sdpa --seed 42 \
  --wandb_project mixture-of-memory --wandb_run_name $RUN" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!  (log: logs/$RUN.log)"
