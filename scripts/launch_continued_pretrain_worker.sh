#!/bin/bash
# Launch continued pretraining on a SINGLE worker node.
# Usage: bash scripts/launch_continued_pretrain_worker.sh <NODE_RANK> [EXTRA_ARGS...]
#
# NODE_RANK: 0=b200-1 (master), 1=b200-2, 2=b200-3, 3=b200-4

set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

NODE_RANK=${1:-0}
shift || true

MASTER_IP="28.89.17.143"
MASTER_PORT=29500
NNODES=4
NPROC_PER_NODE=8

SHARD_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
OUTPUT_DIR="outputs/continued_pretrain_dolmino"
WIKITEXT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy"

torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${MASTER_PORT} \
    --rdzv_id=continued_pretrain \
    --node_rank=${NODE_RANK} \
    scripts/train_continued_pretrain.py \
    --shard_dir ${SHARD_DIR} \
    --model ${MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --wikitext_path ${WIKITEXT} \
    --num_shards 100 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --num_slots 64 \
    --top_k 8 \
    --lora_rank 16 \
    --lr 3e-5 \
    --warmup_steps 200 \
    --max_steps 2000 \
    --gradient_accumulation_steps 4 \
    --eval_interval 200 \
    --save_interval 500 \
    --kl_weight 0.1 \
    $@
