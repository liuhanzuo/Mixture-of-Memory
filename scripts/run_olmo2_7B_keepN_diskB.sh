#!/usr/bin/env bash
# diskB (H20, 97.8GB) launcher for OLMo-2-7B prune-heal (armB = keep-front + fresh) probe#2.
# Single node, 8x H20, torchrun --standalone. Only KEEP / N_FRESH vary; every other
# hyperparam matches the wzc1 keep14 run (clean depth ladder) EXCEPT bs/ga which are
# tuned for H20: 7B fp32-master cannot fit bs16 on 97.8GB, so bs4 ga4 (eff_bs=128,
# identical to keep14/keep10 verified H20 recipe: logs/olmo2_7B_keep10*.log).
#
# All paths/env are diskB defaults but overridable so this same file works on any
# diskB H20 node (.82 / .104). Do NOT hardcode wzc1 paths here.
#
# Usage:  KEEP=12 bash scripts/run_olmo2_7B_keepN_diskB.sh
#         KEEP=8  bash scripts/run_olmo2_7B_keepN_diskB.sh
#         (defaults: KEEP=12 N_FRESH=2 BS=4 GA=4)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

KEEP="${KEEP:-12}"
N_FRESH="${N_FRESH:-2}"
RESUME_FROM="${RESUME_FROM:-}"
BS="${BS:-4}"
GA="${GA:-4}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

DATA_PATH="${DATA_PATH:-/dev/shm/dolmino_now15b.npy}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
OUT_DIR="outputs/olmo2_probe2_7B_keep${KEEP}fresh${N_FRESH}"
LOG_FILE="logs/olmo2_7B_keep${KEEP}fresh${N_FRESH}.log"

mkdir -p "$OUT_DIR" logs

# NCCL: single-node standalone (no cross-node), IB stack broken on these H20 -> TCP.
export NCCL_IB_DISABLE=1
# Remote node cannot reach wandb.ai.
export WANDB_MODE=offline

echo "[run_olmo2_7B_keepN_diskB] KEEP=$KEEP N_FRESH=$N_FRESH BS=$BS GA=$GA eff_bs=$((BS*GA*8)) RESUME_FROM=${RESUME_FROM:-<none>} -> $OUT_DIR (log $LOG_FILE)"

# fresh run starts a clean log; resume APPENDS to preserve pre-resume history.
[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"

nohup "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --keep_front_layers "$KEEP" \
    --n_fresh_layers "$N_FRESH" \
    --batch_size "$BS" \
    --grad_accumulation_steps "$GA" \
    --seq_len 2048 \
    --lr 1e-4 \
    --lr_inherited 2e-5 \
    --max_steps 200000 \
    --gradient_checkpointing 1 \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"} \
  >>"$LOG_FILE" 2>&1 &

echo "[run_olmo2_7B_keepN_diskB] launched pid=$! ; tail -f $LOG_FILE"
