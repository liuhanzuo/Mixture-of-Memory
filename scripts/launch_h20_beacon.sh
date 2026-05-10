#!/bin/bash
# Launch Beacon Qwen-2-7B-Instruct BABILong evaluation on h20-1 (all 6 length bins).
# Beacon ~14GB fp16; 1 GPU per length bin; 6 GPUs (0-5).

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
MODEL=/apdcephfs_zwfy6/share_304376610/pighzliu_code/baselines/beacon-qwen-2-7b-hf
RESULTS=$WORKDIR/babilong_results
NAME=Beacon-Qwen2-7B

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
  LOG=$LOGDIR/babilong_beacon_${L}_$(date +%Y%m%d_%H%M).log
  echo "[launch] gpu=$G length=$L log=$LOG"
  CUDA_VISIBLE_DEVICES=$G nohup $PY $SCRIPT \
    --baseline beacon \
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

echo "Launched 6 Beacon-Qwen2-7B jobs on h20-1 GPUs 0-5"
sleep 3
ps -ef | grep eval_baseline_babilong | grep -v grep | wc -l
