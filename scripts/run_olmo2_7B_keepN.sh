#!/usr/bin/env bash
# Reusable launcher for OLMo-2-7B prune-heal (armB = keep-front + fresh) probe#2.
# Single node, 8x L20A, torchrun --standalone. Only KEEP / N_FRESH vary; every
# other hyperparam is hardcoded to EXACTLY match the keep14 run (clean ablation),
# so keep14 / keep12 / keep10 form a pure depth ladder.
#
# keep14 reference header (logs/olmo2_7B_keep14fresh2.log):
#   world_size=8 bs=16 gaccum=1 eff_bs=128 seq_len=2048 lr_fresh=1e-4 lr_inh=2e-5
#   max_steps=200000 warmup=150 (default) fp32 master weights, grad_checkpointing on.
#
# Usage:  KEEP=12 bash scripts/run_olmo2_7B_keepN.sh
#         (defaults: KEEP=12 N_FRESH=2)
set -euo pipefail

PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"

KEEP="${KEEP:-12}"
N_FRESH="${N_FRESH:-2}"
RESUME_FROM="${RESUME_FROM:-}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# --- checkpoint volume control (added 2026-08-05) -----------------------------
# These arms previously ran at the trainer defaults save_every=500 /
# milestone_every=5000 with max_steps=200000: 400 saves of which the 40
# every-5000 milestones were retained FOREVER -> ~1.8 TB per arm (measured:
# outputs/olmo2_probe2_7B_shortgpt16 = 2.0 TB / 44 ckpts at 46 GB each). The
# unbounded milestone clause -- NOT the latest-N clause -- was the volume driver.
#
# KEEP_MILESTONES caps how many milestones are retained (newest first), so an arm
# is now bounded at roughly KEEP_MILESTONES * 46 GB instead of growing without
# limit. SAVE_EVERY / MILESTONE_EVERY keep their historical values so the retained
# step GRID is unchanged; only the NUMBER of retained milestones is now bounded.
#
# KEEP_STEPS is the escape hatch for load-bearing checkpoints (paired-trajectory
# points, PPL-bracketing points, paper-table rows): they are retained no matter
# what. e.g. KEEP_STEPS=45000,121000 for the keep8 paired-MMLU trajectory.
# step0 is protected automatically.
#
# Set KEEP_LAST_N=0 to DISABLE rotation entirely and keep EVERY save -- required
# for dense-save runs (see scripts/run_olmo2_keep14_densesave_reheal.sh).
SAVE_EVERY="${SAVE_EVERY:-500}"
MILESTONE_EVERY="${MILESTONE_EVERY:-5000}"
KEEP_LAST_N="${KEEP_LAST_N:-3}"
KEEP_MILESTONES="${KEEP_MILESTONES:-8}"
KEEP_STEPS="${KEEP_STEPS:-}"

# --- Paper B control arms (optional, opt-in via env; default = plain depth ladder) ---
# FREEZE_FRONT=1 -> Arm A     : freeze inherited front layers, train fresh+norm+lm_head
# FROM_SCRATCH=1 -> Control 2 : ignore base weights, random-init ALL layers
# Control arms get an ARM-suffixed OUT_DIR/LOG so they never clobber keep14/12/8.
FREEZE_FRONT="${FREEZE_FRONT:-}"
FROM_SCRATCH="${FROM_SCRATCH:-}"
ARM=""
[ -n "$FROM_SCRATCH" ] && ARM="${ARM}_fromscratch"
[ -n "$FREEZE_FRONT" ] && ARM="${ARM}_freezefront"

DATA_PATH="/dev/shm/dolmino_now15b.npy"
MODEL_PATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B"
OUT_DIR="outputs/olmo2_probe2_7B_keep${KEEP}fresh${N_FRESH}${ARM}"
LOG_FILE="logs/olmo2_7B_keep${KEEP}fresh${N_FRESH}${ARM}.log"

mkdir -p "$OUT_DIR" logs

# fp32 master weights is the script DEFAULT (model_dtype hardcoded torch.float32,
# no flag). Hyperparams below are all keep14-identical; only keep_front differs.
echo "[run_olmo2_7B_keepN] KEEP=$KEEP N_FRESH=$N_FRESH ARM=${ARM:-<none>} RESUME_FROM=${RESUME_FROM:-<none>} -> $OUT_DIR (log $LOG_FILE)"
echo "[run_olmo2_7B_keepN] ckpt retention: save_every=$SAVE_EVERY milestone_every=$MILESTONE_EVERY keep_last_n=$KEEP_LAST_N keep_milestones=$KEEP_MILESTONES keep_steps=${KEEP_STEPS:-<none>}"
[ "$KEEP_LAST_N" = "0" ] && echo "[run_olmo2_7B_keepN] NOTE keep_last_n=0 -> ROTATION DISABLED, every save retained"

# fresh run starts a clean log; resume APPENDS to preserve pre-resume history.
[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"

nohup "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --keep_front_layers "$KEEP" \
    --n_fresh_layers "$N_FRESH" \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --seq_len 2048 \
    --lr 1e-4 \
    --lr_inherited 2e-5 \
    --max_steps 200000 \
    --gradient_checkpointing 1 \
    --save_every "$SAVE_EVERY" \
    --milestone_every "$MILESTONE_EVERY" \
    --keep_last_n "$KEEP_LAST_N" \
    --keep_milestones "$KEEP_MILESTONES" \
    ${KEEP_STEPS:+--keep_steps "$KEEP_STEPS"} \
    ${FREEZE_FRONT:+--freeze_front} \
    ${FROM_SCRATCH:+--from_scratch} \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"} \
  >>"$LOG_FILE" 2>&1 &

echo "[run_olmo2_7B_keepN] launched pid=$! ; tail -f $LOG_FILE"
