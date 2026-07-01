#!/usr/bin/env bash
# Watcher for L3 diversity sweep eval (one RUN). Waits for SCHED_DONE + no live harness,
# then scores step500 + step1000 -> logs/<RUN>_scores.txt + DONE_SCORING sentinel.
#   RUN (required), PROJECT_ROOT (default disk A local), PYTHON_BIN (default .venv)
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
: "${RUN:?need RUN}"
LOGDIR=logs/eval_${RUN}
SCORES=logs/${RUN}_scores.txt
while true; do
  live=$(pgrep -fc "run_babilong_mem_space.py" || true)
  if [ -f "$LOGDIR/SCHED_DONE" ] && [ "${live:-0}" -eq 0 ]; then break; fi
  sleep 30
done
{
  echo "=== ${RUN} BABILong scores ($(date)) ==="
  for S in step500 step1000; do
    echo; echo "--- ${RUN}_${S} ---"
    $PYBIN scripts/score_nested_babilong.py "babilong_results/${RUN}_${S}" 2>&1
  done
} > "$SCORES" 2>&1
touch "$LOGDIR/DONE_SCORING"
echo "[$(date)] scoring done -> $SCORES"
