#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python scripts/eval_nih_hard.py \
    --model_path models/Qwen--Qwen3-8b \
    --context_length 65536 \
    --num_needles 15 \
    --num_trials 3 \
    --gpu_id 0 \
    --output_dir outputs/nih_hard_base_qwen3_8b
