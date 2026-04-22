#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/
source .venv/bin/activate
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python scripts/eval_needle_haystack.py \
  --base_model ../models/Qwen--Qwen3-8b \
  --checkpoint outputs/rmt_v10_8gpu_20260419_001626_20260419_001703/final/ \
  --output_dir outputs/nih_eval_v10/ \
  --depths 0.1 0.3 0.5 0.7 0.9 \
  --lengths 1024 2048 4096 \
  --num_trials 3 \
  --seed 42
