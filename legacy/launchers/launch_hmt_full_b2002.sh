#!/bin/bash
set -e

HMT_REPO=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/third_party/HMT-pytorch
MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B
SAVE_DIR=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/hmt_pg19_full_b2002
CACHE_DIR=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/cache/hmt_pg19_full
LOG=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/logs/hmt_pg19_full_b2002.log

mkdir -p "$SAVE_DIR" "$CACHE_DIR" "$(dirname "$LOG")"
mkdir -p "$HMT_REPO/artifact"

cd "$HMT_REPO"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_DATASETS_OFFLINE=1
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export HTTP_PROXY=http://star-proxy.oa.com:3128
export PYTHONPATH="$HMT_REPO:$PYTHONPATH"

nohup /opt/conda/envs/torch-base/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --mixed_precision bf16 \
  tools/training/train_redpajama.py \
  --model_name="$MODEL" \
  --task_name="emozilla/pg19" \
  --batch_size=1 \
  --training_step=50000 \
  --eval_step=1000 \
  --segment_length=512 \
  --num_sensory=32 \
  --use_lora \
  --learning_rate=1e-5 \
  --save_interval=5000 \
  --save_dir="$SAVE_DIR" \
  --cache_dir="$CACHE_DIR" \
  > "$LOG" 2>&1 &

echo "PID: $!"
