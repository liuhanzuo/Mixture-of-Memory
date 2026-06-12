#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

# Patch: run only ctx=1024 to avoid OOM on longer contexts
# Check if eval script supports context_lengths arg
EVALDIR=outputs/eval_nih_slot_memory_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EVALDIR"
python scripts/eval_slot_memory.py \
  --checkpoint_dir outputs/slot_memory_8gpu_stage3_20260420_164731_stage3_20260420_164811/final/ \
  --device cuda:0 \
  --num_trials 100 \
  --output_dir "$EVALDIR"
