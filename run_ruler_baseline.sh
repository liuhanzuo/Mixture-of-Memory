#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/eval_memory_hard_benchmarks.py \
  --model_path models/Qwen--Qwen3-8b/ \
  --benchmarks multi_needle_recall associative_recall passkey_hard counterfactual_retrieval reasoning_chain \
  --context_lengths 8192 16384 32768 \
  --max_context_tokens 40000 \
  --chat_template \
  --output_dir outputs/ruler_baseline_sweep/ \
  --trials 15
