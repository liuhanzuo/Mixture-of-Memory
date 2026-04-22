#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_ppl_v3.py outputs/rmt_v3_lora_20260416_001200/final \
  --heldout data/rmt_train_wiki_zh_10k.jsonl --max_docs 50 \
  > outputs/rmt_v3_lora_20260416_001200/eval_v3.log 2>&1
echo "EVAL_DONE"
