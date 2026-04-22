#!/bin/bash
# NIH-Extended Benchmark: Extended Needle-in-a-Haystack evaluation
#
# Usage:
#   bash scripts/run_eval_nih_extended.sh [MODEL_PATH] [OUTPUT_DIR]
#   bash scripts/run_eval_nih_extended.sh  # defaults: Qwen3-8B base model

set -euo pipefail

cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/
source .venv/bin/activate
export PYTHONUNBUFFERED=1

MODEL_PATH="${1:-../models/Qwen--Qwen3-8b/}"
OUTPUT_DIR="${2:-outputs/nih_extended/}"

echo "=========================================="
echo " NIH-Extended Benchmark"
echo " Model:  $MODEL_PATH"
echo " Output: $OUTPUT_DIR"
echo "=========================================="

# Full eval (~72 tests, ~30-60 min depending on GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_nih_extended.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_trials 3 \
    --max_new_tokens 50 \
    --dtype bfloat16 \
    --seed 42

echo ""
echo "Done! Results in $OUTPUT_DIR"
