#!/usr/bin/env bash
# Launch Beacon-Qwen2-7B BABILong eval across multiple GPUs in parallel.
# Beacon needs ~14-16GB fp16, fits in 1 H20 GPU.

set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

LOGDIR=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs
mkdir -p $LOGDIR

# Use the symlinked path so the activation-beacon trigger in run_model_on_babilong.py fires
MODEL_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/baselines/activation-beacon-qwen2-7b
MODEL_NAME=Beacon-Qwen2-7B-Instruct

# h20-2 GPUs (2-3) for Beacon
declare -A LENGTH_GPU=( [1k]=2 [2k]=3 [4k]=6 [8k]=7 )

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5" "qa6" "qa7" "qa8" "qa9" "qa10")

for length in 1k 2k 4k 8k; do
    gpu=${LENGTH_GPU[$length]}
    LOGFILE=$LOGDIR/babilong_beacon_qwen2_${length}_$(date +%Y%m%d_%H%M).log
    nohup bash scripts/run_babilong_single_h20.sh $MODEL_PATH $MODEL_NAME $gpu $length "${TASKS[@]}" > $LOGFILE 2>&1 &
    disown
    echo "Launched Beacon @ $length on GPU$gpu (PID=$!) -> $LOGFILE"
    sleep 2
done
