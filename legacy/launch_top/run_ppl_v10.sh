#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/eval_rmt.py \
    --checkpoint_dir outputs/rmt_v10_8gpu_20260419_001626_20260419_001703/final/ \
    --data_path data/rmt_train_wiki_zh_10k.jsonl \
    --output_dir outputs/ppl_eval_v10/ \
    --eval_type ppl \
    --max_docs 100
