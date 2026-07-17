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

DATA_PATH="/dev/shm/dolmino_now15b.npy"
MODEL_PATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B"
OUT_DIR="outputs/olmo2_probe2_7B_keep${KEEP}fresh${N_FRESH}"
LOG_FILE="logs/olmo2_7B_keep${KEEP}fresh${N_FRESH}.log"

mkdir -p "$OUT_DIR" logs

# fp32 master weights is the script DEFAULT (model_dtype hardcoded torch.float32,
# no flag). Hyperparams below are all keep14-identical; only keep_front differs.
echo "[run_olmo2_7B_keepN] KEEP=$KEEP N_FRESH=$N_FRESH RESUME_FROM=${RESUME_FROM:-<none>} -> $OUT_DIR (log $LOG_FILE)"

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
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"} \
  >>"$LOG_FILE" 2>&1 &

echo "[run_olmo2_7B_keepN] launched pid=$! ; tail -f $LOG_FILE"
