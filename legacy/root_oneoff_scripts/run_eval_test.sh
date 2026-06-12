#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/eval_slot_memory.py \
  --checkpoint_dir outputs/slot_memory_8gpu_stage3_20260420_164731_stage3_20260420_164811/final \
  --base_model ../models/Qwen--Qwen3-8b \
  --num_trials 3 \
  --device cuda:0 \
  --output_dir eval_results/slot_memory_test
