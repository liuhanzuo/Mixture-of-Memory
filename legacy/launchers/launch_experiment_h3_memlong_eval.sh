#!/bin/bash
# Experiment H3: Middle-layer memory + MemLong-protocol NIAH eval enabled.
# Same architecture as H (write=L16, read={18,22,26,30}), but during training
# we additionally run the MemLong NIAH protocol every 500 steps to get numbers
# directly comparable to the MemLong baseline on b200-4.
#
# Requires b200-4 (or any free 8-GPU node) — pick whichever is free at launch.
set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h3_memlong_eval_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H3: middle-layer memory + MemLong NIAH"
echo "  Write layer: 16"
echo "  Read layers: 18,22,26,30"
echo "  MemLong-NIAH eval every 500 steps"
echo "Output: outputs/experiment_h3_memlong_eval"
echo "Log: $LOGFILE"
echo "============================================"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train_cross_attn_memory.py \
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h3_memlong_eval \
    --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
    --niah_data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
    --slot_forward \
    --middle_layer_memory \
    --memory_write_layer 16 \
    --memory_read_layers 18,22,26,30 \
    --memory_init strided \
    --niah_mix_fraction 0.30 \
    --niah_max_N 2 \
    --niah_warmup_steps 100 \
    --lambda_retrieve 1.0 \
    --memlong_niah_interval 500 \
    --memlong_niah_lengths 2048,4096 \
    --memlong_niah_depths 0.0,0.25,0.5,0.75,1.0 \
    --memlong_niah_trials 3 \
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
    --max_steps 5000 \
    --eval_interval 200 \
    --save_interval 1000 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    2>&1 | tee $LOGFILE
