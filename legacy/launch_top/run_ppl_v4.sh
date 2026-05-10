#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
python scripts/quick_ppl.py outputs/rmt_v4_simple_8gpu_10ep/final data/rmt_train_wikitext.jsonl 20 > outputs/rmt_v4_simple_8gpu_10ep/eval/quick_ppl_v4.log 2>&1
