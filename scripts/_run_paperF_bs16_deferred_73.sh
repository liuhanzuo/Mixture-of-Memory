#!/usr/bin/env bash
# Wait for Paper B training jobs to finish on .73, then run bs16 eval for paper F.
# Safe: polls every 10 min. Only starts eval when GPU memory is free.
# Idempotent: the inner eval script skips already-done rungs.
set -u

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
LOG=logs/paperF_bs16_ladder_deferred.log

echo "[$(date '+%F %T')] Watchdog started: waiting for .73 training to finish" | tee -a "$LOG"
echo "Will poll every 10 minutes for GPU memory free (< 5000 MiB per card)" | tee -a "$LOG"

while true; do
    # Check if training processes are still using GPU memory
    MAX_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if [ "$MAX_MEM" -lt 5000 ]; then
        echo "[$(date '+%F %T')] GPU memory free (max=${MAX_MEM} MiB). Starting bs16 eval." | tee -a "$LOG"
        break
    fi
    echo "[$(date '+%F %T')] GPU busy (max=${MAX_MEM} MiB). Sleeping 10 min..." | tee -a "$LOG"
    sleep 600
done

# Run the bs16 eval (idempotent -- skips already-done rungs)
bash scripts/_run_paperF_bs16_ladder_73.sh >> "$LOG" 2>&1
echo "[$(date '+%F %T')] bs16 ladder eval done. Running final analysis..." | tee -a "$LOG"

# Run analysis
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
/opt/conda/envs/torch-base/bin/python scripts/paperF_bs_ladder_analysis.py >> "$LOG" 2>&1
echo "[$(date '+%F %T')] Analysis done. Check status/PAPERF_BS_LADDER_VERDICT.md on wzc1." | tee -a "$LOG"
