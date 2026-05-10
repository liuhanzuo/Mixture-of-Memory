#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export PYTHONUNBUFFERED=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/rmt_v8_8gpu_${TIMESTAMP}"
echo $OUTPUT_DIR > outputs/v8_output_dir.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 scripts/train_rmt_v5.py \
  --data data/rmt_train_wiki_zh_10k.jsonl \
  --base_model ../models/Qwen--Qwen3-8b \
  --output_dir "$OUTPUT_DIR" \
  --num_memory_tokens 64 \
  --segment_length 1024 \
  --max_segments 6 \
  --num_epochs 20 \
  --lr 2e-5 \
  --rmt_lr 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --warmup_steps 50 \
  --bottleneck_dim 256 \
  --extractor_version 5 \
  --log_every 10 \
  --save_every 500 \
  --ddp \
  --seed 42

echo "Training completed. Output: $OUTPUT_DIR"
