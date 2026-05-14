#!/bin/bash
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs outputs/full_finetune_nomem
torchrun --nproc_per_node=8 scripts/train_full_finetune.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
  --num_shards 25 --shard_offset 25 \
  --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
  --full_finetune --no_memory \
  --seq_len 4096 --chunks_per_doc 32 \
  --lr 5e-6 --warmup_steps 200 --max_steps 2000 \
  --gradient_checkpointing --gradient_accumulation_steps 4 \
  --kl_weight 0.0 --eval_interval 100 --save_interval 500 \
  --output_dir outputs/full_finetune_nomem
