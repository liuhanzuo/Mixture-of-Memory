#!/bin/bash
# NIH Extended Benchmark — Base + RMT v10
# Uses checkpoint with proper v10 weights (l0.memory keys)
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/
source .venv/bin/activate
export PYTHONUNBUFFERED=1

BASE_MODEL="../models/Qwen--Qwen3-8b"
CHECKPOINT="${1:-outputs/rmt_v10_20260419_182044/final/}"
OUTPUT_DIR="${2:-outputs/nih_extended_v10/}"

echo "=== NIH Extended Base + RMT v10 ==="
echo "Base: $BASE_MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python scripts/eval_needle_haystack_extended.py \
  --base_model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --modes single multi \
  --lengths 4096 8192 16384 \
  --depths 0.1 0.25 0.5 0.75 0.9 \
  --multi_needles 3 5 \
  --num_trials 3 \
  --seed 42 \
  --lang en
