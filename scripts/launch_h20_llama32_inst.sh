#!/bin/bash
# Launch Llama-3.2-1B-Instruct BABILong evaluation on h20-2 across multiple GPUs.
# Each GPU runs one length bin in parallel (small model = 2-3 GB, fits easily).

set -u
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export HF_DATASETS_CACHE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/hf_cache/datasets
export PYTHONUNBUFFERED=1
mkdir -p $HF_DATASETS_CACHE

WORKDIR=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
LOGDIR=$WORKDIR/logs
mkdir -p $LOGDIR
cd $WORKDIR

PY=/opt/conda/envs/torch-base/bin/python
SCRIPT=$WORKDIR/scripts/eval_baseline_babilong.py
MODEL=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Llama-3.2-1B-Instruct
RESULTS=$WORKDIR/babilong_results
NAME=Llama-3.2-1B-Instruct

LENGTHS_GPU=(
  "1k:0"
  "2k:1"
  "4k:2"
  "8k:3"
  "16k:4"
  "32k:5"
)

for entry in "${LENGTHS_GPU[@]}"; do
  L="${entry%%:*}"
  G="${entry##*:}"
  LOG=$LOGDIR/babilong_llama32_inst_${L}_$(date +%Y%m%d_%H%M).log
  echo "[launch] gpu=$G length=$L log=$LOG"
  CUDA_VISIBLE_DEVICES=$G nohup $PY $SCRIPT \
    --baseline plain_hf \
    --model_path $MODEL \
    --output_name $NAME \
    --results_folder $RESULTS \
    --tasks qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10 \
    --lengths $L \
    --use_chat_template --use_instruction --use_examples --use_post_prompt \
    --max_new_tokens 20 \
    --limit 100 \
    > $LOG 2>&1 &
  disown
  sleep 1
done

echo "Launched 6 Llama-3.2-1B-Instruct jobs on h20-2 GPUs 0-5"
ps -ef | grep eval_baseline_babilong | grep -v grep | head
