#!/bin/bash
# Experiment H2: Deeper middle-layer memory (ablation vs H)
# Purpose: Test sensitivity of write_layer position — write at L20, read at {22,25,28,31}
# Hypothesis: Is "middle" ≈ L/2 optimal (MemLong-style), or does deeper (~L*5/8) give higher-level semantics?
# Compare: H (write=16, read=18,22,26,30) vs H2 (write=20, read=22,25,28,31)
# Target: Remote b200-3 (28.89.17.85), 8× GPU
set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h2_deeper_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H2: Deeper Middle-Layer Memory"
echo "  Write layer: 20 (vs H's 16)"
echo "  Read layers: 22,25,28,31 (vs H's 18,22,26,30)"
echo "Output: outputs/experiment_h2_deeper"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h2_deeper \
    --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
    --niah_data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
    --slot_forward \
    --middle_layer_memory \
    --memory_write_layer 20 \
    --memory_read_layers 22,25,28,31 \
    --memory_init strided \
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
    --max_steps 5000 \
    --eval_interval 200 \
    --save_interval 1000 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    2>&1 | tee $LOGFILE
