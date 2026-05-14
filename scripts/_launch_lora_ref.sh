#!/bin/bash
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs outputs/lora_best_ref
torchrun --nproc_per_node=8 scripts/train_full_finetune.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
  --num_shards 25 --shard_offset 0 \
  --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
  --lora_finetune --lora_rank 16 --use_memory \
  --seq_len 4096 --chunks_per_doc 32 \
  --num_slots 64 --top_k 8 \
  --lr 1e-4 --warmup_steps 100 --max_steps 2000 \
  --gradient_checkpointing --gradient_accumulation_steps 4 \
  --kl_weight 0.0 --eval_interval 100 --save_interval 500 \
  --output_dir outputs/lora_best_ref
