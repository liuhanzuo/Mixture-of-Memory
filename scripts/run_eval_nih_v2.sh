#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

python scripts/eval_nih_multisegment_v2.py \
  --model_path models/Qwen--Qwen3-8b \
  --stage2_path outputs/slot_memory_8gpu_stage2_20260420_122501_stage2_20260420_122538/final \
  --stage3_path outputs/slot_memory_8gpu_stage3_20260420_164731_stage3_20260420_164811/final \
  --output_dir outputs/nih_multisegment \
  --device cuda:0 \
  --num_trials 3 \
  --segment_length 1024
