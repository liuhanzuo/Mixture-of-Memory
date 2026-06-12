#!/usr/bin/env bash
# Launch one ROUTE-A arm's full eval (driver) + scoring watcher, both fully
# detached via setsid, with their own log files. Invoke this itself under setsid
# so nothing is tied to the ssh/agent session. Idempotent-ish: refuses if a
# driver for this ARM is already running.
#
# Required env: ARM, CKPT_DIR
# Optional env: PROJECT_ROOT, PYTHON_BIN, GPUS
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
: "${ARM:?must set ARM}"
: "${CKPT_DIR:?must set CKPT_DIR}"
export WANDB_MODE=offline
mkdir -p logs

if pgrep -af "[p]ython scripts/run_babilong_mem_space.py" | grep -q "output_name ${ARM}_step"; then
  echo "[$(date)] driver for $ARM already has eval procs running; not relaunching" \
    >> logs/launch_${ARM}.log
  exit 0
fi

# Driver: detached.
setsid env ARM="$ARM" CKPT_DIR="$CKPT_DIR" \
  PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="${PYTHON_BIN:-}" GPUS="${GPUS:-0 1 2 3 4 5 6 7}" \
  bash scripts/eval_routeA_arm_2ckpt.sh </dev/null >logs/eval_${ARM}_driver.log 2>&1 &
echo "[$(date)] driver launched pid $! ARM=$ARM" >> logs/launch_${ARM}.log

sleep 3
# Watcher: detached.
setsid env ARM="$ARM" PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="${PYTHON_BIN:-}" \
  bash scripts/watch_score_routeA_arm.sh </dev/null >logs/watch_${ARM}.log 2>&1 &
echo "[$(date)] watcher launched pid $! ARM=$ARM" >> logs/launch_${ARM}.log
