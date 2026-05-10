#!/usr/bin/env bash
# Launches Llama-3.2-1B-Instruct BABILong eval across multiple GPUs in parallel on h20-2.
# Each GPU handles one (length) bin; all 10 tasks within that bin.
# Designed to start as a background job (nohup + disown).

set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

LOGDIR=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs
mkdir -p $LOGDIR

MODEL_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Llama-3.2-1B-Instruct
MODEL_NAME=Llama-3.2-1B-Instruct

# h20-2 GPUs 4-7 used for Llama-3.2-1B-Instruct
# 1k -> GPU4, 2k -> GPU5, 4k -> GPU6, 8k -> GPU7
declare -A LENGTH_GPU=( [1k]=4 [2k]=5 [4k]=6 [8k]=7 )

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5" "qa6" "qa7" "qa8" "qa9" "qa10")

for length in 1k 2k 4k 8k; do
    gpu=${LENGTH_GPU[$length]}
    LOGFILE=$LOGDIR/babilong_llama32_instruct_${length}_$(date +%Y%m%d_%H%M).log
    nohup bash scripts/run_babilong_single_h20.sh $MODEL_PATH $MODEL_NAME $gpu $length "${TASKS[@]}" > $LOGFILE 2>&1 &
    disown
    echo "Launched $MODEL_NAME @ $length on GPU$gpu (PID=$!) -> $LOGFILE"
    sleep 2
done
