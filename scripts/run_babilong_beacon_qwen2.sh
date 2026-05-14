#!/usr/bin/env bash
# BABILong-100 evaluation: Activation Beacon (Qwen2-7B based)
# Reference: arXiv:2401.03462, namespace-Pt/beacon-qwen-2-7b-instruct
# Compression-style memory; uses interleaved beacon tokens.

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/babilong

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONUNBUFFERED=1

LOGDIR=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/babilong_beacon_qwen2_7b_$(date +%Y%m%d_%H%M).log

RESULTS_FOLDER=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/babilong_results
MODEL_NAME="Beacon-Qwen2-7B-Instruct"
MODEL_PATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/baselines/beacon-qwen-2-7b
DATASET_NAME="RMT-team/babilong"

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

GPU=${1:-6}
echo "Launching Beacon eval on GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU /opt/conda/envs/torch-base/bin/python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    --use_chat_template \
    --use_instruction \
    --use_examples \
    --use_post_prompt \
    --api_url "" \
    2>&1 | tee $LOGFILE
