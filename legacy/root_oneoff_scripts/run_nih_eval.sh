#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/eval_slot_memory.py \
  --checkpoint_dir outputs/slot_memory_8gpu_stage3_20260420_164731_stage3_20260420_164811/final/ \
  --device cuda:0 \
  --num_trials 5 \
  2>&1
