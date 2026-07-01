#!/usr/bin/env bash
# Robust full-node waiter: poll until ALL 8 local GPUs are free (<5GiB used),
# then launch expL1KEY_indepkey_N128. Prior waiter fired early -> OOM (11:28).
# Require 3 consecutive clean polls (90s) to avoid race with dying eval procs.
set -uo pipefail
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
LOG=logs/waiter_expL1KEY_relaunch.log
echo "$(date '+%F %T') waiter armed; polling for all-8-GPU-free" > "$LOG"
clean=0
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>5000{c++} END{print c+0}')
  if [ "$busy" -eq 0 ]; then
    clean=$((clean+1))
    echo "$(date '+%F %T') all GPUs free (clean=$clean/3)" >> "$LOG"
    if [ "$clean" -ge 3 ]; then break; fi
  else
    clean=0
    echo "$(date '+%F %T') $busy GPU(s) still busy, waiting" >> "$LOG"
  fi
  sleep 30
done
echo "$(date '+%F %T') launching expL1KEY_indepkey_N128" >> "$LOG"
bash scripts/launch_expL1KEY_indepkey_N128.sh >> "$LOG" 2>&1
echo "$(date '+%F %T') launch script returned" >> "$LOG"
