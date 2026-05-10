#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/
source .venv/bin/activate
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python scripts/eval_rmt.py \
  --checkpoint_dir outputs/rmt_v9_8gpu_20260418_083226_20260418_083302/final/ \
  --data_path data/squad_val.jsonl \
  --eval_type nih \
  --nih_num_trials 1 \
  --output_dir eval_results/rmt_v9_fixed_inference/
