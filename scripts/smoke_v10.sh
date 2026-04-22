#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rmt_v10.py \
  --data data/rmt_train_wiki_zh_10k.jsonl \
  --val_data data/rmt_train_wiki_zh_10k.jsonl \
  --base_model ../models/Qwen--Qwen3-8b \
  --output_dir outputs/smoke_v10 \
  --num_memory_tokens 64 \
  --segment_length 1024 \
  --max_segments 2 \
  --num_epochs 1 \
  --lr 2e-5 \
  --rmt_lr 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --warmup_steps 1 \
  --bottleneck_dim 256 \
  --extractor_version 5 \
  --log_every 1 \
  --lambda_retrieve 0.5 \
  --retrieve_every 1 \
  --seed 42 \
  --wandb_project ""
