#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/eval_rmt.py \
  --checkpoint_dir outputs/rmt_v4_simple_8gpu_10ep/final/ \
  --data_path data/rmt_train_mixed.jsonl \
  --eval_type ppl \
  --num_memory_tokens 16 \
  --bottleneck_dim 64 \
  --segment_length 1024 \
  --max_segments 6 \
  --output_dir eval_results/rmt_v4_10ep \
  --max_docs 100
