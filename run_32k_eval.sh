#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/eval_nih_extended_sparse.py \
  --model_path ../models/Qwen--Qwen3-8b/ \
  --output_dir outputs/sparse_32k_full_eval_20260421 \
  --device cuda:0 \
  --config_filter "32768" \
  --num_trials 3 \
  --seed 42 \
  2>&1 | tee outputs/sparse_32k_full_eval_20260421.log
