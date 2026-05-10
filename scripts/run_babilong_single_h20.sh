#!/usr/bin/env bash
# BABILong-100 launcher: runs run_model_on_babilong.py for one task/length combination on a single GPU.
# Usage: run_one.sh <model_path> <model_name> <gpu> <length> [tasks...]
#
# Designed for H20 cluster (NVIDIA H20 97.8 GB). Targets a single GPU; allows parallel launches.

set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/babilong

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONUNBUFFERED=1

MODEL_PATH=$1
MODEL_NAME=$2
GPU=$3
LENGTH=$4
shift 4
TASKS=("$@")

LOGDIR=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/babilong_${MODEL_NAME}_${LENGTH}_$(date +%Y%m%d_%H%M).log

RESULTS_FOLDER=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/babilong_results
DATASET_NAME="RMT-team/babilong"

EXTRA_FLAGS=""
case "$MODEL_NAME" in
    Llama-3.2-1B-Instruct|*Instruct*|*chat*)
        EXTRA_FLAGS="--use_chat_template --use_instruction --use_examples --use_post_prompt"
        ;;
    *)
        EXTRA_FLAGS="--use_instruction --use_examples --use_post_prompt"
        ;;
esac

echo "============================================" | tee -a $LOGFILE
echo "BABILong-100 baseline launcher" | tee -a $LOGFILE
echo "  Model name: $MODEL_NAME" | tee -a $LOGFILE
echo "  Model path: $MODEL_PATH" | tee -a $LOGFILE
echo "  GPU: $GPU" | tee -a $LOGFILE
echo "  Length: $LENGTH" | tee -a $LOGFILE
echo "  Tasks: ${TASKS[@]}" | tee -a $LOGFILE
echo "  Extra flags: $EXTRA_FLAGS" | tee -a $LOGFILE
echo "============================================" | tee -a $LOGFILE

CUDA_VISIBLE_DEVICES=$GPU /opt/conda/envs/torch-base/bin/python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --tasks "${TASKS[@]}" \
    --lengths "$LENGTH" \
    $EXTRA_FLAGS \
    --api_url "" \
    2>&1 | tee -a $LOGFILE
