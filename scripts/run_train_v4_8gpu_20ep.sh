#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:29502 scripts/train_rmt_v4.py \
  --data data/rmt_train_mixed.jsonl \
  --base_model ../models/Qwen--Qwen3-8b \
  --output_dir outputs/rmt_v4_8gpu_20ep \
  --num_memory_tokens 16 \
  --segment_length 1024 \
  --max_segments 6 \
  --num_epochs 20 \
  --lr 2e-5 \
  --rmt_lr 5e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --warmup_steps 30 \
  --log_every 10 \
  --save_every 100 \
  --bottleneck_dim 64 \
  --recon_weight 0.1 \
  --extractor_version 3 \
  --seed 42
