#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/eval_nih_extended.py \
  --model_path ../models/Qwen--Qwen3-8b/ \
  --output_dir outputs/sparse_memory_nih_extended_eval_fixed \
  --device cuda:0 \
  --smoke 2>&1
