#!/bin/bash
# Contrastive Retrieval Loss experiment for NIAH training.
# Adds InfoNCE loss supervising query->slot attention during NIAH forward pass.
#
# Abort criteria (logged in code):
# - If NIAH accuracy still 0% at step 1000 and niah_loss > 5.5 -> experiment failed
# - If PPL ratio > 1.05 -> contrastive loss hurting LM quality
#
# Backward compatible: contrastive_weight=0.0 = original behavior (no contrastive loss).

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

torchrun --nproc_per_node=8 scripts/train_cross_attn_memory.py \
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
    --output_dir outputs/contrastive_retrieval \
    --wikitext_path data/wikitext_chunks_llama3_4096.npy \
    --niah_data data/pg19_chunks_llama3.npy \
    --niah_mix_fraction 0.30 \
    --niah_max_N 2 \
    --niah_warmup_steps 100 \
    --lambda_retrieve 1.0 \
    --num_shards 25 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --num_slots 64 \
    --top_k 8 \
    --use_cross_attn_memory \
    --residual_scale 1.0 \
    --write_lr 0.1 \
    --cross_attn_lr_factor 10 \
    --contrastive_weight 1.0 \
    --contrastive_temperature 0.1 \
    --full_finetune \
    --use_memory \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 5e-6 \
    --min_lr 1e-6 \
    --warmup_steps 200 \
    --max_steps 2000 \
    --eval_interval 100 \
    --save_interval 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0
