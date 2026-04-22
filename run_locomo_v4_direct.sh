#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/eval_rmt_locomo.py \
  --checkpoint_dir outputs/rmt_v4_simple_8gpu_10ep \
  --locomo_data locomo/data/locomo10.json \
  --output_dir eval_results/locomo_v4 \
  --max_new_tokens 50 \
  --num_memory_tokens 16 \
  --segment_length 1024 \
  --max_segments 6 \
  --bottleneck_dim 64 \
  --extractor_version 2 \
  2>&1 | tee eval_results/locomo_v4.log
