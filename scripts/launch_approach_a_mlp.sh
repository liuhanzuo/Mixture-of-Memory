#!/bin/bash
# Approach A: MLP Compression Write Head (--memory_init=mlp)
# MLP compresses mean-pooled hidden states into memory slots (explicit "write" mechanism)
# No freeze/unfreeze — train everything jointly from step 0
# Target: b200-1 (28.89.17.143), 8 GPU
set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/approach_a_mlp_$(date +%Y%m%d_%H%M).log

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train_cross_attn_memory.py \
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/approach_a_mlp \
    --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
    --niah_data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
    --slot_forward \
    --memory_init mlp \
    --niah_mix_fraction 0.30 \
    --niah_max_N 2 \
    --niah_warmup_steps 100 \
    --lambda_retrieve 1.0 \
    --num_shards 25 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --num_slots 64 \
    --full_finetune \
    --use_memory \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 5e-6 \
    --min_lr 1e-6 \
    --warmup_steps 200 \
    --max_steps 2000 \
    --eval_interval 200 \
    --save_interval 1000 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    2>&1 | tee $LOGFILE
