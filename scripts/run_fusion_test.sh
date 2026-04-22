#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
python scripts/validate_fusion.py
