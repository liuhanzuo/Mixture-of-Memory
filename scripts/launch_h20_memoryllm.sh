#!/bin/bash
# Launch MemoryLLM-8B-chat BABILong evaluation across H20 nodes.
# 6 lengths total. Pass NODE=h20-1 or h20-2 to control which node + which GPUs.

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
MODEL=/apdcephfs_zwfy6/share_304376610/pighzliu_code/baselines/memoryllm-8b-chat-hf
RESULTS=$WORKDIR/babilong_results
NAME=MemoryLLM-8B-chat

# Caller passes "L1:G1 L2:G2 ..." as args
if [ $# -eq 0 ]; then
  echo "usage: $0 1k:6 2k:7 4k:0 ..."
  exit 1
fi

for entry in "$@"; do
  L="${entry%%:*}"
  G="${entry##*:}"
  LOG=$LOGDIR/babilong_memoryllm_${L}_$(date +%Y%m%d_%H%M).log
  echo "[launch] gpu=$G length=$L log=$LOG"
  CUDA_VISIBLE_DEVICES=$G nohup $PY $SCRIPT \
    --baseline memoryllm \
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

sleep 3
echo "Launched MemoryLLM jobs:"
ps -ef | grep eval_baseline_babilong | grep memoryllm | grep -v grep | wc -l
