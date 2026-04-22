#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export PYTHONUNBUFFERED=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 scripts/train_slot_memory.py \
  --data data/rmt_train_wiki_zh_10k.jsonl \
  --base_model ../models/Qwen--Qwen3-8b \
  --output_dir outputs/slot_memory_8gpu_${TIMESTAMP} \
  --num_slots 16 \
  --slot_dim 256 \
  --segment_length 1024 \
  --max_segments 6 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --warmup_steps 30 \
  --log_every 10 \
  --ddp \
  --seed 42

# === Single-GPU debug variant ===
# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# CUDA_VISIBLE_DEVICES=0 python scripts/train_slot_memory.py \
#   --data data/rmt_train_wiki_zh_10k.jsonl \
#   --base_model ../models/Qwen--Qwen3-8b \
#   --output_dir outputs/slot_memory_debug_${TIMESTAMP} \
#   --num_slots 16 \
#   --slot_dim 256 \
#   --segment_length 1024 \
#   --max_segments 6 \
#   --num_epochs 1 \
#   --lr 2e-5 \
#   --lora_r 16 \
#   --lora_alpha 32 \
#   --lora_dropout 0.05 \
#   --batch_size 1 \
#   --grad_accumulation_steps 8 \
#   --warmup_steps 30 \
#   --log_every 5 \
#   --val_every 100 \
#   --save_every 100 \
#   --wandb_project slot-memory \
#   --seed 42
