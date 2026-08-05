#!/usr/bin/env bash
# QCMem self-distillation (Direction A, 2026-07-05) — Qwen3-8B, PG19 pure LM.
# Teacher = QCMem read at j=0 (RAG upper bound, adapters OFF).
# Student = QCMem read at j=RESUME_J (default 12) + LoRA r16 on layers[j:].
# Loss = bidirectional top-64 KL on the QUERY-segment tokens. PG19 natural text
# only (NO babilong / NO needles / NO eval data — red line).
#
# Goal: push back the mid-depth-resume depth cliff on precise-localisation tasks
# (qa1 j12=11 zero-training) toward the j0 teacher, mirroring the QCMem paper's
# LoRA self-distillation. This is the PURE-PG19 arm: whether it suffices for Qwen
# (the paper needed a needle-mix on Llama) is the open question.
#
# === USAGE ===================================================================
# Local B200 (8x L20A, wzc1) — .venv torch2.10:
#   PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory \
#   setsid nohup bash scripts/launch_qcmem_distill_qwen_pg19.sh \
#     >logs/qcmem_distill_qwen_pg19.out 2>&1 &
# Override defaults via env: RESUME_J=15 N_CTX=7 TOTAL_STEPS=1500 LR=1e-4 ...
# ==============================================================================
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # set in your shell env (see CLAUDE.md); NOT hardcoded
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

MODEL="${MODEL:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b}"
RESUME_J="${RESUME_J:-12}"
TOP_PREPAY_B="${TOP_PREPAY_B:-0}"
LORA_RANK="${LORA_RANK:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
N_CTX="${N_CTX:-7}"                 # (7+1)*512 = 4096-tok training window
TEACHER_TOPK="${TEACHER_TOPK:-64}"
DISTILL_LAMBDA="${DISTILL_LAMBDA:-0.6}"
CE_WEIGHT="${CE_WEIGHT:-0.0}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-50}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29971}"
RUN="${RUN:-qcmem_distill_qwen_j${RESUME_J}b${TOP_PREPAY_B}_pg19_nctx${N_CTX}}"
OUTPUT_DIR="outputs/$RUN"
mkdir -p logs "$OUTPUT_DIR"

echo "[launch] RUN=$RUN model=$MODEL j=$RESUME_J b=$TOP_PREPAY_B n_ctx=$N_CTX steps=$TOTAL_STEPS nproc=$NPROC"

setsid bash -c "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
  $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$PORT \
  scripts/train_qcmem_distill.py \
  --model_path $MODEL \
  --resume_j $RESUME_J --top_prepay_b $TOP_PREPAY_B \
  --lora_rank $LORA_RANK --chunk_size $CHUNK_SIZE --n_ctx $N_CTX \
  --teacher_topk $TEACHER_TOPK --distill_lambda $DISTILL_LAMBDA --ce_weight $CE_WEIGHT \
  --total_steps $TOTAL_STEPS --lr $LR --warmup_steps $WARMUP --grad_accum $GRAD_ACCUM \
  --gradient_checkpointing \
  --output_dir $OUTPUT_DIR --save_interval 250 --log_interval 10 \
  --keep_last_n ${KEEP_LAST_N:-3} --keep_steps "${KEEP_STEPS:-1000,2000,3000,4000}" \
  --dtype bfloat16 --attn_impl sdpa --seed 42 \
  --wandb_project mixture-of-memory --wandb_run_name $RUN" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!  (log: logs/$RUN.log)"
