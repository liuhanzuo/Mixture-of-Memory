#!/usr/bin/env bash
# Self-launching L3-diversity eval driver for ONE run.
# 1) Wait for step1000 ckpt (mem_space_adapter.pt) to appear.
# 2) Wait for the training node's GPUs to free up (no train procs for this output_dir).
# 3) Run the jobpool eval (step500+step1000) then score -> logs/<RUN>_scores.txt + DONE_SCORING.
# Designed to be launched detached via setsid.
#   RUN, CKPT_DIR (required); PROJECT_ROOT, PYTHON_BIN, GPUS, TRAIN_TAG optional.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
: "${RUN:?need RUN}"; : "${CKPT_DIR:?need CKPT_DIR}"
TRAIN_TAG="${TRAIN_TAG:-$RUN}"   # substring to detect this run's training procs
LOGDIR=logs/eval_${RUN}; mkdir -p "$LOGDIR"
DRIVER_LOG="$LOGDIR/driver.log"
exec >>"$DRIVER_LOG" 2>&1

echo "[$(date)] DRIVER start RUN=$RUN CKPT_DIR=$CKPT_DIR TRAIN_TAG=$TRAIN_TAG"

# 1) wait for step1000 final ckpt
while [ ! -f "${CKPT_DIR}/mem_space_adapter.pt" ]; do
  echo "[$(date)] waiting step1000 ckpt..."; sleep 60
done
echo "[$(date)] step1000 ckpt present"

# 2) wait for training procs of this run to exit (GPUs free)
while pgrep -fa "train_mem_space_dolmino_cpt.py" 2>/dev/null | grep -q -- "$TRAIN_TAG"; do
  echo "[$(date)] training still running ($TRAIN_TAG), waiting..."; sleep 60
done
echo "[$(date)] training procs gone, GPUs free -> launch jobpool"

# 3) run jobpool (blocks until SCHED_DONE) then score
RUN="$RUN" CKPT_DIR="$CKPT_DIR" PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}" GPUS="${GPUS:-0 1 2 3 4 5 6 7}" \
  bash scripts/eval_l3div_jobpool.sh

RUN="$RUN" PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}" \
  bash scripts/eval_l3div_watch.sh
echo "[$(date)] DRIVER done RUN=$RUN"
