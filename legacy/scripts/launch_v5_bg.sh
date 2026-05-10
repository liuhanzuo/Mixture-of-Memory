#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 scripts/train_rmt_v5.py \
  --data data/rmt_train_mixed.jsonl \
  --base_model ../models/Qwen--Qwen3-8b \
  --output_dir outputs/rmt_v5_8gpu_20260416_204830 \
  --num_memory_tokens 16 \
  --segment_length 1024 \
  --max_segments 6 \
  --num_epochs 3 \
  --lr 2e-5 \
  --rmt_lr 5e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --warmup_steps 30 \
  --bottleneck_dim 256 \
  --extractor_version 5 \
  --log_every 1 \
  --lambda_zforce 0.1 \
  --seed 42 \
  --ddp \
  2>&1 | tee outputs/rmt_v5_train.log
