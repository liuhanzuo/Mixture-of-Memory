#!/usr/bin/env bash
# BABILong-100 evaluation: Llama-3.2-1B base baseline
#
# Same-size no-memory comparison for LM2 (which is built on Llama-3.2-1B).
# This isolates the memory module's contribution from raw model capability.

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/babilong

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONUNBUFFERED=1

LOGDIR=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/babilong_llama3.2_1b_baseline_$(date +%Y%m%d_%H%M).log

RESULTS_FOLDER=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/babilong_results
MODEL_NAME="Meta-Llama-3.2-1B"
MODEL_PATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B
DATASET_NAME="RMT-team/babilong"

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

echo "============================================"
echo "BABILong-100 baseline: $MODEL_NAME"
echo "  Path: $MODEL_PATH"
echo "  Tasks: ${TASKS[@]}"
echo "  Lengths: ${LENGTHS[@]}"
echo "============================================"

CUDA_VISIBLE_DEVICES=4 /opt/conda/envs/torch-base/bin/python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    --use_instruction \
    --use_examples \
    --use_post_prompt \
    --api_url "" \
    2>&1 | tee $LOGFILE
