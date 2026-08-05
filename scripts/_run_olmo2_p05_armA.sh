#!/usr/bin/env bash
# Paper B P0.5 structural-isolation control -- Arm A: contiguous16 / no-fresh.
# Keeps the ORIGINAL front 16 base OLMo-2 layers (0..15) with NO freshly-init tail
# (n_fresh=0, pure ShortGPT-style transplant + heal), so ALL trainable params heal
# at the single inherited LR 2e-5. This is the "keep-a-contiguous-prefix, no tail
# re-grow" endpoint of the structural-isolation experiment; Arm B (retained
# [0..12,31] + 2 fresh) is the paired non-contiguous + fresh-tail arm.
#
# Reuses scripts/train_olmo2_shortgpt.py verbatim (no code change needed: it already
# transplants an arbitrary --keep_layer_indices set with n_fresh hardwired to 0).
#
# ⚠️ DRY-by-default: prints the exact launch command and only EXECUTES when RUN=1.
# The operator / MAIN starts the 200k heal deliberately on a FREE 8-GPU node.
#
# Recipe (identical to _run_olmo2_shortgpt_heal.sh, the keep14-matched heal):
#   seq_len 2048, eff batch 128 (H20: BS=4 GA=4 nproc=8), max_steps 200000,
#   cosine, warmup 150, wd 0.1, grad_clip 1.0, save_every 5000, gc 1, seed 42,
#   fp32 master weights, NCCL_IB_DISABLE=1, WANDB offline, expandable_segments.
#
# Usage:
#   bash scripts/_run_olmo2_p05_armA.sh          # DRY: print the command
#   RUN=1 bash scripts/_run_olmo2_p05_armA.sh    # actually launch (8 GPU)
#   RESUME_FROM=outputs/olmo2_p05_armA_contig16/step5000.pt RUN=1 bash ... # resume
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/olmo2_venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-../models/OLMo-2-1124-7B}"
DATA_PATH="${DATA_PATH:-/dev/shm/dolmino_now15b.npy}"
OUT_DIR="${OUT_DIR:-outputs/olmo2_p05_armA_contig16}"
LOG_FILE="${LOG_FILE:-logs/olmo2_p05_armA_contig16.log}"
RESUME_FROM="${RESUME_FROM:-}"
NPROC="${NPROC:-8}"
# H20 (97.8GB) cannot fit bs16 fp32-master 7B -> BS=4 GA=4 (eff 128). B200 -> BS=16 GA=1.
BS="${BS:-4}"
GA="${GA:-4}"
SEED="${SEED:-42}"
KEEP_INDICES="${KEEP_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

mkdir -p "$OUT_DIR" logs

# NCCL / offline wandb / allocator (match the diskB + wzc1 healed-arm recipe).
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Build the launch command as a bash array (avoid quoting/line-continuation pitfalls).
CMD=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC"
  scripts/train_olmo2_shortgpt.py
    --data_path "$DATA_PATH"
    --output_dir "$OUT_DIR"
    --model_path "$MODEL_PATH"
    --keep_layer_indices "$KEEP_INDICES"
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
    --extra_save_steps 50000,100000,150000
    # ckpt rotation (bound the 200k-step volume; extra_save_steps + step0
    # remain protected). KEEP_LAST_N=0 disables rotation entirely.
    --keep_last_n ${KEEP_LAST_N:-3}
    --keep_milestones ${KEEP_MILESTONES:-8}
    --gradient_checkpointing 1
    --seed "$SEED"
)
[ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")

echo "[_run_olmo2_p05_armA] keep_layer_indices=$KEEP_INDICES (n_fresh=0, contiguous16) seed=$SEED"
echo "[_run_olmo2_p05_armA] eff_bs=$((BS*GA*NPROC)) -> $OUT_DIR (log $LOG_FILE)"
echo "----- launch command -----"
printf '  %q' "${CMD[@]}"; echo
echo "--------------------------"

if [ "${RUN:-0}" != "1" ]; then
  echo "[_run_olmo2_p05_armA] DRY RUN (set RUN=1 to launch the 200k heal on 8 GPU)."
  exit 0
fi

# fresh run starts a clean log; resume appends.
[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"
echo "[_run_olmo2_p05_armA] LAUNCHING 8-GPU heal ..."
nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
echo "[_run_olmo2_p05_armA] launched pid=$! ; tail -f $LOG_FILE"
