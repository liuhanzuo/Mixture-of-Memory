#!/usr/bin/env bash
# Detached watcher: wait until all eval procs for d6nullsinkoff drain, then score both ckpts.
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=.venv/bin/python
OUT=logs/d6nullsinkoff_scores.txt
: > "$OUT"
echo "[watcher] start $(date)" >> "$OUT"

# wait for scheduler + eval procs to drain (allow startup grace)
sleep 30
while true; do
  n=$(ps aux | grep -E "run_babilong_mem_space.py|_d6nullsinkoff_eval_scheduler.py" | grep -v grep | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "[watcher] all procs drained at $(date)" >> "$OUT"
    break
  fi
  sleep 30
done

for run in d6nullsinkoff_step500 d6nullsinkoff_step5000; do
  echo "==================== $run ====================" >> "$OUT"
  $PY scripts/score_nested_babilong.py "babilong_results/$run" >> "$OUT" 2>&1
  echo "" >> "$OUT"
done

echo "[watcher] DONE_SCORING $(date)" >> "$OUT"
touch logs/d6nullsinkoff_DONE_SCORING
