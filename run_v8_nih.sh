#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/eval_nih.py \
  --checkpoint_dir outputs/rmt_v8_8gpu_20260418_011145_20260418_011221/final/ \
  --output_dir outputs/nih_eval_v8/
