#!/usr/bin/env bash
# 8-GPU heal launch driver for the ShortGPT prune-then-heal arm (Paper B external
# baseline). Reads the ShortGPT layer selection, keeps those 16 layers (no fresh
# tail, n_fresh=0), and heals on Dolmino with EXACTLY the keep14 recipe: seq_len
# 2048, effective batch 128, gradient checkpointing, fp32 master weights,
# max_steps 200000, cosine schedule, warmup 150. ALL params use the inherited LR
# 2e-5 (no fresh-LR bucket, since ShortGPT re-grows no tail).
#
# ⚠️ This script does NOT auto-start the long run: it prints the exact launch
# command and (only if RUN=1) executes it. Default is DRY (print only), so a
# heartbeat / operator starts the 200k heal deliberately on a free 8-GPU node.
#
# Prereqs:
#   * outputs/shortgpt_layer_selection.json exists (scripts/_run_shortgpt_select.sh)
#   * /dev/shm/dolmino_now15b.npy staged (same as keep14; else set DATA_PATH)
#
# Usage:
#   bash scripts/_run_olmo2_shortgpt_heal.sh            # DRY: print the command
#   RUN=1 bash scripts/_run_olmo2_shortgpt_heal.sh      # actually launch (8 GPU)
#   RESUME_FROM=outputs/olmo2_probe2_7B_shortgpt16/step5000.pt RUN=1 bash ... # resume
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
DATA_PATH="${DATA_PATH:-/dev/shm/dolmino_now15b.npy}"
SELECTION_JSON="${SELECTION_JSON:-outputs/shortgpt_layer_selection.json}"
OUT_DIR="${OUT_DIR:-outputs/olmo2_probe2_7B_shortgpt16}"
LOG_FILE="${LOG_FILE:-logs/olmo2_7B_shortgpt16.log}"
RESUME_FROM="${RESUME_FROM:-}"
NPROC="${NPROC:-8}"
# H20 (97.8GB) cannot fit bs16 fp32-master 7B -> BS=4 GA=4; B200 (183GB) -> BS=16 GA=1.
BS="${BS:-16}"
GA="${GA:-1}"

mkdir -p "$OUT_DIR" logs

if [ ! -f "$SELECTION_JSON" ]; then
  echo "[_run_olmo2_shortgpt_heal] ERROR: $SELECTION_JSON missing. Run scripts/_run_shortgpt_select.sh first." >&2
  exit 1
fi

# NCCL / offline wandb / allocator (match the diskB + wzc1 healed-arm recipe).
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Build the launch command as a bash array to avoid any quoting/line-continuation
# pitfalls (an earlier heredoc+`bash -c "$CMD"` form fed literal backslashes to
# argparse as spurious arguments).
CMD=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC"
  scripts/train_olmo2_shortgpt.py
    --data_path "$DATA_PATH"
    --output_dir "$OUT_DIR"
    --model_path "$MODEL_PATH"
    --selection_json "$SELECTION_JSON"
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
    --extra_save_steps 128000,153500
    # ckpt rotation: bound this 200k-step run's volume. Previously the 40
    # every-5000 milestones were retained forever (~1.8 TB). extra_save_steps
    # + step0 stay protected regardless. keep_last_n 0 would disable rotation.
    --keep_last_n ${KEEP_LAST_N:-3}
    --keep_milestones ${KEEP_MILESTONES:-8}
    --gradient_checkpointing 1
)
[ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")

echo "[_run_olmo2_shortgpt_heal] selection=$SELECTION_JSON keep=$(${PYTHON_BIN} -c "import json;print(json.load(open('$SELECTION_JSON'))['kept_layer_indices'])" 2>/dev/null)"
echo "[_run_olmo2_shortgpt_heal] eff_bs=$((BS*GA*NPROC)) -> $OUT_DIR (log $LOG_FILE)"
echo "----- launch command -----"
printf '  %q' "${CMD[@]}"; echo
echo "--------------------------"

if [ "${RUN:-0}" != "1" ]; then
  echo "[_run_olmo2_shortgpt_heal] DRY RUN (set RUN=1 to launch the 200k heal on 8 GPU)."
  exit 0
fi

# fresh run starts a clean log; resume appends.
[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"
echo "[_run_olmo2_shortgpt_heal] LAUNCHING 8-GPU heal ..."
nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
echo "[_run_olmo2_shortgpt_heal] launched pid=$! ; tail -f $LOG_FILE"
