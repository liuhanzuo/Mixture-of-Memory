#!/usr/bin/env bash
# Sharded BABILong long-context (16k+32k) baseline for Meta-Llama-3-8B-Instruct
# (no memory) on the second H20 node (28.59.80.196).
# gp-7 variant: writes to babilong_results/Meta-Llama-3-8B-Instruct-node2/ to
# avoid clobbering the node-1 results dir (which holds 0k-8k for the same
# model from a prior run).  6 shards on GPUs 0..5, GPUs 6,7 left idle.
set -e
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)/third_party/babilong-pkg:$(pwd):$PYTHONPATH

PYBIN=/opt/conda/envs/torch-base/bin/python
TS=$(date +%Y%m%d_%H%M)
OUTPUT_NAME=Meta-Llama-3-8B-Instruct-node2

mkdir -p logs babilong_results/$OUTPUT_NAME

GPU=0
PIDS=()
for TASK in qa1 qa2 qa5; do
  for LEN in 16k 32k; do
    LOG=logs/node2_8b_instruct_${TASK}_${LEN}_${TS}.log
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYBIN scripts/eval_baseline_babilong.py \
      --baseline plain_hf \
      --model_path models/Meta-Llama-3-8B-Instruct \
      --output_name $OUTPUT_NAME \
      --results_folder ./babilong_results \
      --dataset_name RMT-team/babilong \
      --tasks $TASK --lengths $LEN \
      --use_chat_template --use_instruction --use_examples --use_post_prompt \
      --max_new_tokens 20 --limit 100 \
      > $LOG 2>&1 &
    P=$!
    PIDS+=($P)
    echo "Launched $TASK/$LEN on GPU $GPU, PID $P, log $LOG"
    GPU=$((GPU+1))
    sleep 2
  done
done
echo "All 6 shards launched: ${PIDS[*]}"
