#!/usr/bin/env bash
# Wait for the mem_space eval scheduler to finish, then score all 5 runs.
set -u
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
OUT=logs/eval_scores_final.txt

# Block until scheduler process is gone AND no worker procs remain.
while pgrep -f "run_mem_space_eval_shards" > /dev/null || \
      [ "$(pgrep -af 'run_babilong_mem_space' | grep -v ssh | wc -l)" -gt 0 ]; do
  sleep 60
done

{
  echo "=== ALL EVAL UNITS FINISHED: $(date) ==="
  echo "scheduler log tail:"
  tail -8 logs/eval_scheduler.log
  echo
  for run in f2_c512_step500 f2_c512_step5000 ladder_s1c256_step500 ladder_s1c256_step5000 ladder_s2c512_step500; do
    echo "########## $run ##########"
    PYTHONPATH=third_party/babilong-pkg .venv/bin/python scripts/score_nested_babilong.py "babilong_results/$run" 2>&1
    echo
  done
} > "$OUT" 2>&1
echo "DONE_SCORING" >> "$OUT"
