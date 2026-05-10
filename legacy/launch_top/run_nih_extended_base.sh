#!/bin/bash
# NIH Extended Benchmark — Base model only (fast, no RMT)
# Tests 4K, 8K, 16K, 32K to find where base model degrades
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/
source .venv/bin/activate
export PYTHONUNBUFFERED=1

BASE_MODEL="${1:-../models/Qwen--Qwen3-8b}"
OUTPUT_DIR="${2:-outputs/nih_extended_base/}"

echo "=== NIH Extended Base-Only ==="
echo "Base: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python scripts/eval_needle_haystack_extended.py \
  --base_model "$BASE_MODEL" \
  --base_only \
  --output_dir "$OUTPUT_DIR" \
  --modes single multi \
  --lengths 4096 8192 16384 32768 \
  --depths 0.1 0.25 0.5 0.75 0.9 \
  --multi_needles 3 5 \
  --num_trials 5 \
  --seed 42 \
  --lang en
