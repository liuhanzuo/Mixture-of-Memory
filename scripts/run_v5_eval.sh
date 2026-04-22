#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 python scripts/eval_rmt.py \
  --checkpoint_dir outputs/rmt_v5_8gpu_20260416_234601/final \
  --data_path data/rmt_train_mixed.jsonl \
  --output_dir outputs/rmt_v5_8gpu_20260416_234601/eval \
  --eval_type ppl \
  --num_memory_tokens 16 \
  --segment_length 1024 \
  --bottleneck_dim 256 \
  --max_docs 200
