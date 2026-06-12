#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
python scripts/eval_memory_hard_benchmarks.py \
  --model_path models/Qwen--Qwen3-8b \
  --benchmarks multi_needle_recall associative_recall reasoning_chain counterfactual_retrieval passkey_hard \
  --context_lengths 16384 32768 65536 131072 \
  --trials 5 \
  --output_dir outputs/hard_benchmarks/baseline_qwen3_8b \
  --max_context_tokens 131072 \
  --seed 42 \
  > outputs/hard_benchmarks/baseline_qwen3_8b/eval_full.log 2>&1
