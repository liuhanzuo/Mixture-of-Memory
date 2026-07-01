#!/usr/bin/env bash
# Idempotent launcher for the topk8 step500 eval watchdog. Invoked over SSH.
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
n=$(ps -eo pid,args | grep -E "[b]ash scripts/watch_eval_p8b_topk8_step500.sh" | wc -l)
if [ "$n" -gt 0 ]; then echo "ALREADY_RUNNING n=$n"; exit 0; fi
setsid bash scripts/watch_eval_p8b_topk8_step500.sh </dev/null >logs/watch_eval_p8b_topk8_step500.nohup 2>&1 &
echo "STARTED"
