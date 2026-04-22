#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
OUTPUT_DIR="outputs/sparse_memory_nih_extended_eval_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=1 python scripts/eval_nih_extended_sparse.py \
    --model_path ../models/Qwen--Qwen3-8b/ \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --num_trials 3 \
    2>&1
