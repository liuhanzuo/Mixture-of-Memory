#!/bin/bash
set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NODE_NAME=${1:-"unknown"}
LR=${2:-"3e-5"}
LORA_RANK=${3:-"16"}
SHARD_OFFSET=${4:-"0"}
OUTPUT_NAME=${5:-"baseline"}

mkdir -p logs "outputs/continued_pretrain_${OUTPUT_NAME}"

echo "Launching ${OUTPUT_NAME} on ${NODE_NAME} at $(date)"
echo "LR=${LR}, LORA_RANK=${LORA_RANK}, SHARD_OFFSET=${SHARD_OFFSET}"

torchrun --nproc_per_node=8 scripts/train_continued_pretrain.py \
    --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --output_dir "outputs/continued_pretrain_${OUTPUT_NAME}" \
    --wikitext_path data/wikitext_chunks_llama3_4096.npy \
    --num_shards 25 --shard_offset ${SHARD_OFFSET} \
    --seq_len 4096 --chunks_per_doc 32 \
    --num_slots 64 --top_k 8 --lora_rank ${LORA_RANK} \
    --lr ${LR} --warmup_steps 100 --max_steps 2000 \
    --gradient_accumulation_steps 1 \
    --eval_interval 100 --save_interval 500 --kl_weight 0 \
    2>&1 | tee "logs/continued_pretrain_${NODE_NAME}_${TIMESTAMP}.log"
