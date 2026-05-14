#!/usr/bin/env bash
# BABILong-100 evaluation: Llama-3-8B baseline (vanilla, no memory module)
#
# Reference: https://github.com/booydar/babilong
# Dataset: RMT-team/babilong (100 samples per task per length)
# Tasks: qa1-qa5 (single/multi-fact retrieval)
# Lengths: 0k/1k/2k/4k/8k/16k/32k
#
# Target: b200-2 (single GPU is enough; 8B fp16 fits on one L20A 183GB)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/babilong

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONUNBUFFERED=1

LOGDIR=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/babilong_llama3_8b_baseline_$(date +%Y%m%d_%H%M).log

RESULTS_FOLDER=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/babilong_results
MODEL_NAME="Meta-Llama-3-8B"
MODEL_PATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATASET_NAME="RMT-team/babilong"

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

echo "============================================"
echo "BABILong-100 baseline: $MODEL_NAME"
echo "  Path: $MODEL_PATH"
echo "  Tasks: ${TASKS[@]}"
echo "  Lengths: ${LENGTHS[@]}"
echo "  Output: $RESULTS_FOLDER/$MODEL_NAME"
echo "  Log:    $LOGFILE"
echo "============================================"

# Llama-3-8B is a base model (not Instruct), so we DON'T use chat template
# but DO use instruction + examples + post_prompt to give it task structure.
# This matches the babilong repo's style for non-instruct LLaMA-2 evals.

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/torch-base/bin/python scripts/run_model_on_babilong.py \
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
