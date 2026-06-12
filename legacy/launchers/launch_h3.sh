#!/bin/bash
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

mkdir -p outputs/experiment_h3_aggressive_niah

/opt/conda/envs/torch-base/bin/python3.11 /opt/conda/envs/torch-base/bin/torchrun \
  --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
  scripts/train_cross_attn_memory.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
  --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h3_aggressive_niah \
  --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
  --niah_data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
  --slot_forward --middle_layer_memory \
  --memory_write_layer 16 --memory_read_layers 18,22,26,30 \
  --memory_init strided \
  --niah_mix_fraction 0.50 \
  --niah_max_N 2 --niah_warmup_steps 100 \
  --lambda_retrieve 3.0 \
  --num_shards 25 --seq_len 4096 --chunks_per_doc 32 \
  --num_slots 128 \
  --full_finetune --use_memory \
  --gradient_checkpointing --gradient_accumulation_steps 4 \
  --lr 5e-6 --min_lr 1e-6 --warmup_steps 200 --max_steps 5000 \
  --eval_interval 200 --save_interval 1000 \
  --weight_decay 0.01 --max_grad_norm 1.0 \
  > outputs/experiment_h3_aggressive_niah/train.log 2>&1
