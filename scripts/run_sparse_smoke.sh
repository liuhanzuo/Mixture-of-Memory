#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/eval_nih_extended_sparse.py \
    --model_path ../models/Qwen--Qwen3-8b/ \
    --output_dir outputs/sparse_memory_nih_extended_eval_smoke/ \
    --smoke \
    --device cuda:0 \
    2>&1
