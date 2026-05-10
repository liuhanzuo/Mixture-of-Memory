#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train_rmt_v3.py \
  --model_path ../models/Qwen--Qwen3-8b \
  --data_path data/rmt_train_mixed.jsonl \
  --output_dir outputs/rmt_v3_lora_20260416_001200 \
  --num_memory_tokens 8 --segment_length 1024 --max_segments 6 \
  --num_epochs 3 --lr 2e-5 --rmt_lr 5e-5 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --batch_size 1 --grad_accumulation_steps 8 \
  --warmup_steps 30 --log_every 10 --bottleneck_dim 32
