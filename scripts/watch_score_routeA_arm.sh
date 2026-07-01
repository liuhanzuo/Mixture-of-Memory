#!/usr/bin/env bash
# Detached scoring watcher for one ROUTE-A arm. Blocks until the arm's eval
# processes (run_babilong_mem_space.py for this ARM) drop to zero, then scores
# both step500 + step1000 nested result dirs and writes a single scores file +
# DONE_SCORING marker. Run under setsid nohup so it survives ssh/agent recycling.
#
# Required env: ARM (e.g. routeA_arm1)
# Optional env: PROJECT_ROOT, PYTHON_BIN, EVAL_PID (driver pid to wait on)
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
: "${ARM:?must set ARM}"
SCORES=logs/${ARM}_scores.txt
MARKER=logs/${ARM}_DONE_SCORING
mkdir -p logs

echo "[$(date)] watcher start for $ARM" > "$SCORES"

# Wait until no run_babilong_mem_space.py process references this ARM's output dirs.
while true; do
  N=$(pgrep -af "[p]ython scripts/run_babilong_mem_space.py" | grep -c "output_name ${ARM}_step" || true)
  if [ "$N" -eq 0 ]; then
    # Double-check after a short settle to avoid racing a between-units gap.
    sleep 30
    N2=$(pgrep -af "[p]ython scripts/run_babilong_mem_space.py" | grep -c "output_name ${ARM}_step" || true)
    [ "$N2" -eq 0 ] && break
  fi
  sleep 60
done

echo "[$(date)] eval procs zero for $ARM, scoring..." >> "$SCORES"
for STEP in step500 step1000; do
  RD=babilong_results/${ARM}_${STEP}
  echo "" >> "$SCORES"
  echo "########## ${ARM}_${STEP} ##########" >> "$SCORES"
  if [ -d "$RD" ]; then
    $PYBIN scripts/score_nested_babilong.py "$RD" >> "$SCORES" 2>>"$SCORES"
  else
    echo "MISSING result dir $RD" >> "$SCORES"
  fi
done
echo "[$(date)] DONE scoring $ARM" >> "$SCORES"
date > "$MARKER"
