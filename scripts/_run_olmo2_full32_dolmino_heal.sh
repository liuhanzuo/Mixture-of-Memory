#!/usr/bin/env bash
# Full-32L continued-pretraining control (Paper B #1 reviewer control).
# NO pruning: keep ALL 32 pretrained OLMo-2-7B layers (keep_front=32, n_fresh=0),
# transplant every layer + embed/norm/lm_head from base, and continue-train on
# Dolmino with EXACTLY the keep14 recipe: seq_len 2048, effective batch 128,
# gradient checkpointing, fp32 master weights, max_steps 200000, cosine schedule,
# warmup 150. ALL params use LR 2e-5 (both --lr and --lr_inherited set to 2e-5,
# so lm_head's "fresh" bucket also runs at 2e-5 -> a single uniform 2e-5 recipe).
#
# Answers the top reviewer question: is the MMLU collapse from LAYER PRUNING, or
# from Dolmino continued-pretraining itself (corpus-induced forgetting)? If
# full32@200k MMLU stays ~.60, pruning is the cause; if it also drops, the claim
# must be re-framed as pruning + distribution-shift.
#
# ⚠️ DRY by default: prints the launch command, executes only if RUN=1. Start the
# 200k run deliberately on a free 8-GPU node (B200 183GB -> BS16 GA1; H20 97.8GB
# cannot fit bs16 fp32-master full-32L -> use BS4 GA4).
#
# Usage:
#   bash scripts/_run_olmo2_full32_dolmino_heal.sh            # DRY: print command
#   RUN=1 bash scripts/_run_olmo2_full32_dolmino_heal.sh      # launch (8 GPU)
#   RESUME_FROM=outputs/olmo2_probe2_7B_full32_dolmino/step5000.pt RUN=1 bash ...  # resume
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
DATA_PATH="${DATA_PATH:-/dev/shm/dolmino_now15b.npy}"
OUT_DIR="${OUT_DIR:-outputs/olmo2_probe2_7B_full32_dolmino}"
LOG_FILE="${LOG_FILE:-logs/olmo2_7B_full32_dolmino.log}"
RESUME_FROM="${RESUME_FROM:-}"
NPROC="${NPROC:-8}"
# B200 (183GB) -> BS=16 GA=1 (eff_bs 128). H20 (97.8GB) full-32L fp32-master -> BS=4 GA=4.
BS="${BS:-16}"
GA="${GA:-1}"

mkdir -p "$OUT_DIR" logs

if [ ! -f "$DATA_PATH" ]; then
  echo "[_run_olmo2_full32_dolmino_heal] ERROR: $DATA_PATH missing. Stage Dolmino npy (same as keep14) or set DATA_PATH." >&2
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC"
  scripts/train_olmo2_arch_probe2.py
    --data_path "$DATA_PATH"
    --output_dir "$OUT_DIR"
    --model_path "$MODEL_PATH"
    --keep_front_layers 32
    --n_fresh_layers 0
    --lr 2e-5
    --min_lr 2e-6
    --lr_inherited 2e-5
    --min_lr_inherited 2e-6
    --batch_size "$BS"
    --grad_accumulation_steps "$GA"
    --seq_len 2048
    --max_steps 200000
    --warmup_steps 150
    --weight_decay 0.1
    --grad_clip 1.0
    --save_every 5000
    --gradient_checkpointing 1
)
[ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")

echo "[_run_olmo2_full32_dolmino_heal] full-32L (no prune) uniform LR 2e-5 eff_bs=$((BS*GA*NPROC)) -> $OUT_DIR (log $LOG_FILE)"
echo "----- launch command -----"
printf '  %q' "${CMD[@]}"; echo
echo "--------------------------"

if [ "${RUN:-0}" != "1" ]; then
  echo "[_run_olmo2_full32_dolmino_heal] DRY RUN (set RUN=1 to launch the 200k control on 8 GPU)."
  exit 0
fi

[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"
echo "[_run_olmo2_full32_dolmino_heal] LAUNCHING 8-GPU full-32L continued-pretraining control ..."
nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
echo "[_run_olmo2_full32_dolmino_heal] launched pid=$! ; tail -f $LOG_FILE"
