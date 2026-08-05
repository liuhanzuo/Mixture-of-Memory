#!/usr/bin/env bash
# QCMem WRITE-PATH self-distillation (Paper A P1.10, 2026-08-03) — Qwen3-8B, PG19.
#
# Trains a LoRA on the LOWER j=12 layers (indices 0..11 = the Write path) so the
# deployable CHUNK-LOCAL Write learns to emit a document-contextual h12, distilled
# against the P0.18 E0 "closes-to-100" teacher (continuous lower-12 over the whole
# packed window). The flagship READ LoRA (layers 12..35) is merged into the base as a
# frozen Read, shared by teacher & student — only the Write path differs.
#
# CONTRAST vs the flagship READ-path trainer (scripts/train_qcmem_distill.py):
#   flagship: LoRA on layers[12:36] (Read), Write frozen & under no_grad.
#   this:     LoRA on layers[0:12]  (Write), Read frozen (merged) — grad flows
#             through the frozen Read back into the Write LoRA.
#
# === USAGE (node .104, 8x H20, diskB torch-base) =============================
#   PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash scripts/launch_qcmem_writepath_distill_diskB.sh \
#     >logs/qcmem_writepath_distill.out 2>&1 &
# Short multi-GPU sanity (finite decreasing loss, no OOM/NaN) before the full run:
#   MAX_STEPS_SMOKE=30 TOTAL_STEPS=4000 ... bash scripts/launch_..._diskB.sh
# Override defaults via env: BATCH_SIZE=8 LR=8e-5 TOTAL_STEPS=4000 ...
# ==============================================================================
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

MODEL="${MODEL:-models/Qwen3-8b-local}"
READ_LORA="${READ_LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
RESUME_J="${RESUME_J:-12}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
N_CTX="${N_CTX:-3}"                 # (3+1)*512 = 2048-tok window
BATCH_SIZE="${BATCH_SIZE:-8}"       # windows/step (fixed length -> batched, no pad)
TEACHER_TOPK="${TEACHER_TOPK:-64}"
DISTILL_LAMBDA="${DISTILL_LAMBDA:-0.6}"
CE_WEIGHT="${CE_WEIGHT:-0.0}"
TOTAL_STEPS="${TOTAL_STEPS:-4000}"
LR="${LR:-8e-5}"
WARMUP="${WARMUP:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS_SMOKE="${MAX_STEPS_SMOKE:-0}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29973}"
RUN="${RUN:-qcmem_writepath_distill_qwen_j${RESUME_J}_r${LORA_RANK}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$RUN}"
mkdir -p logs "$OUTPUT_DIR"

echo "[launch] RUN=$RUN model=$MODEL read_lora=$READ_LORA j=$RESUME_J "\
"n_ctx=$N_CTX batch=$BATCH_SIZE steps=$TOTAL_STEPS lr=$LR smoke=$MAX_STEPS_SMOKE nproc=$NPROC pybin=$PYBIN"

setsid bash -c "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
  $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$PORT \
  scripts/train_qcmem_writepath_distill.py \
  --model_path $MODEL --read_lora_path $READ_LORA \
  --resume_j $RESUME_J --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \
  --chunk_size $CHUNK_SIZE --n_ctx $N_CTX --batch_size $BATCH_SIZE \
  --teacher_topk $TEACHER_TOPK --distill_lambda $DISTILL_LAMBDA --ce_weight $CE_WEIGHT \
  --total_steps $TOTAL_STEPS --lr $LR --warmup_steps $WARMUP --grad_accum $GRAD_ACCUM \
  --gradient_checkpointing --max_steps_smoke $MAX_STEPS_SMOKE \
  --output_dir $OUTPUT_DIR --save_interval 500 --log_interval 10 \
  --keep_last_n ${KEEP_LAST_N:-3} --keep_steps "${KEEP_STEPS:-1000,1500,2000}" \
  --dtype bfloat16 --attn_impl sdpa --seed 42 \
  --wandb_project mixture-of-memory --wandb_run_name $RUN" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!  (log: logs/$RUN.log)"
