#!/usr/bin/env bash
# Detached watcher for routeA_arm4 eval: waits for the scheduler to finish
# (SCHED_DONE sentinel + no live run_babilong processes), then scores both
# ckpts and writes logs/routeA_arm4_scores.txt + DONE_SCORING.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
PYBIN="$PROJECT_ROOT/.venv/bin/python"
LOGDIR="logs/routeA_arm4_eval"
SCORES="logs/routeA_arm4_scores.txt"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"

# Wait for scheduler sentinel AND no live harness procs (belt + suspenders).
while true; do
  live=$(pgrep -fc "run_babilong_mem_space.py" || true)
  if [ -f "$LOGDIR/SCHED_DONE" ] && [ "${live:-0}" -eq 0 ]; then
    break
  fi
  sleep 30
done

{
  echo "=== routeA_arm4 BABILong scores ($(date)) ==="
  for run in routeA_arm4_step500 routeA_arm4_step1000; do
    echo
    $PYBIN scripts/score_nested_babilong.py "babilong_results/$run" 2>&1
  done
} > "$SCORES" 2>&1
touch "$LOGDIR/DONE_SCORING"
echo "[$(date)] scoring done -> $SCORES"
