#!/bin/bash
set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONUNBUFFERED=1
export PYTHONPATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH
python3 scripts/eval_attention_matching.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
  --max_chunks 10 \
  --seq_len 4096 \
  --compression_ratios 2,4,8,16 \
  --output_file results/attention_matching_eval.json
