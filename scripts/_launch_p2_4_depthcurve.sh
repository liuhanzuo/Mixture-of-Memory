#!/usr/bin/env bash
# ============================================================================
# Paper A P2.4 — distilled multi-depth quality-latency curve.
#
# Trains 3 rank-32 LoRA adapters that reproduce the flagship QCMem
# self-distillation recipe EXACTLY (verbatim from the flagship
# outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json), changing ONLY the
# student resume depth j from 12 to {6, 9, 18}. This extends the "depth is a
# tunable reuse axis" thesis from the j=0/12 two-point comparison into a curve.
#
#   Flagship recipe (qcmem_distill_qwen_j12_r32_4k, GROUND TRUTH):
#     backbone       = Qwen3-8B (36 layers)
#     teacher        = QCMem read at j=0 (RAG upper bound, adapters OFF, no grad)
#     student        = QCMem read at j=RESUME_J + LoRA r32/a64 on layers[j:36]
#     lora_targets   = q,k,v,o,gate,up,down (all 7 linear projections)
#     data           = PG19 natural text (data/pg19_train.jsonl), n_ctx=3,
#                      chunk=512 -> (3+1)*512 = 2048-tok packed windows
#     loss           = bidirectional top-64 KL (distill_lambda=0.6, ce_weight=0)
#     total_steps    = 4000   lr=8e-5  warmup=100  wd=0  grad_accum=1  grad_clip=1
#     grad_ckpt      = FALSE (matches flagship)
#     dtype=bf16  attn=sdpa  seed=42  save_interval=500  log_interval=10
#     parallelism    = 8-GPU DDP (eff batch = 8 windows/step)
#
# Data order / seed / eff-batch are identical to the flagship because the PG19
# jsonl is byte-identical across disks, seed=42, world_size=8, n_ctx=3 all match.
# Only the LoRA layer span (and hence trainable-param count) differs per depth —
# this is expected and is NOT compute-matched (documented in the report).
#
# The 3 depths run SERIALLY (one 8-GPU DDP job at a time) so eff batch is fixed.
#
# === USAGE (on .82 diskB, torch-base) =======================================
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash scripts/_launch_p2_4_depthcurve.sh \
#     >logs/p2_4_depthcurve_master.log 2>&1 &
# Override depths: DEPTHS="6 9 18"   (default). GRAD_CKPT=1 enables gradient
# checkpointing as an OOM safety net (gradient-identical; default off = flagship).
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODEL="${MODEL:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b}"
NPROC="${NPROC:-8}"
DEPTHS="${DEPTHS:-6 9 18}"

# ---- flagship recipe knobs (verbatim from distill_args.json; DO NOT CHANGE) --
TOP_PREPAY_B=0
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.0
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
CHUNK_SIZE=512
N_CTX=3
QUERY_LOSS_TOKENS=0
TEACHER_TOPK=64
DISTILL_LAMBDA=0.6
CE_WEIGHT=0.0
TOTAL_STEPS=4000
LR=8e-05
WARMUP=100
WEIGHT_DECAY=0.0
GRAD_ACCUM=1
GRAD_CLIP=1.0
SAVE_INTERVAL=500
LOG_INTERVAL=10
DTYPE=bfloat16
ATTN_IMPL=sdpa
SEED=42
# --------------------------------------------------------------------------- #

# gradient checkpointing off by default (== flagship). Set GRAD_CKPT=1 to enable
# as an OOM safety net (mathematically gradient-identical, does not affect the
# result, data order, seed or eff batch).
GC_FLAG=""
if [[ "${GRAD_CKPT:-0}" == "1" ]]; then
  GC_FLAG="--gradient_checkpointing"
fi

mkdir -p logs

echo "[p2.4] PROJECT_ROOT=$PROJECT_ROOT"
echo "[p2.4] PYBIN=$PYBIN"
echo "[p2.4] MODEL=$MODEL"
echo "[p2.4] depths (serial) = $DEPTHS   gc_flag='$GC_FLAG'"
echo "[p2.4] flagship recipe: r32/a64 n_ctx3 chunk512 steps4000 lr8e-5 warmup100 seed42 8-GPU DDP"
echo "[p2.4] START $(date -u +%FT%TZ)"

# distinct base master port per node (bumped per depth to avoid stale-zombie
# port reuse across the serial arms).
BASE_PORT="${BASE_PORT:-29981}"
i=0
for J in $DEPTHS; do
  RUN="qcmem_distill_qwen_j${J}_r32_4k"
  OUTPUT_DIR="outputs/$RUN"
  LOG="logs/p2_4_distill_j${J}.log"
  PORT=$(( BASE_PORT + i ))
  mkdir -p "$OUTPUT_DIR"
  echo "----------------------------------------------------------------------"
  echo "[p2.4] === depth j=$J -> $OUTPUT_DIR  (port $PORT, log $LOG) ==="
  echo "[p2.4] arm start $(date -u +%FT%TZ)"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  "$PYBIN" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$PORT" \
    scripts/train_qcmem_distill.py \
    --model_path "$MODEL" \
    --resume_j "$J" --top_prepay_b "$TOP_PREPAY_B" \
    --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --lora_targets "$LORA_TARGETS" \
    --pg19_path "$PROJECT_ROOT/data/pg19_train.jsonl" \
    --chunk_size "$CHUNK_SIZE" --n_ctx "$N_CTX" --query_loss_tokens "$QUERY_LOSS_TOKENS" \
    --teacher_topk "$TEACHER_TOPK" --distill_lambda "$DISTILL_LAMBDA" --ce_weight "$CE_WEIGHT" \
    --total_steps "$TOTAL_STEPS" --lr "$LR" --warmup_steps "$WARMUP" \
    --weight_decay "$WEIGHT_DECAY" --grad_accum "$GRAD_ACCUM" --grad_clip "$GRAD_CLIP" \
    $GC_FLAG \
    --output_dir "$OUTPUT_DIR" --save_interval "$SAVE_INTERVAL" --log_interval "$LOG_INTERVAL" \
    --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" --seed "$SEED" \
    --wandb_project mixture-of-memory --wandb_run_name "" \
    >"$LOG" 2>&1
  rc=$?
  echo "[p2.4] arm j=$J finished rc=$rc  $(date -u +%FT%TZ)"
  if [[ $rc -ne 0 ]]; then
    echo "[p2.4] WARNING: depth j=$J exited non-zero (rc=$rc). See $LOG. Continuing to next depth."
  fi
  i=$(( i + 1 ))
done

echo "[p2.4] ALL DEPTHS DONE $(date -u +%FT%TZ)"
