#!/bin/bash
# Scheme A: Cross-Attention Memory -- 2-arm ablation
# Arm 1 (b200-2): num_slots=64, top_k=8
# Arm 2 (b200-3): num_slots=128, top_k=16
#
# Usage:
#   bash scripts/launch_cross_attn_memory.sh <ARM> [NODE_RANK]
#   ARM: arm1_64 | arm2_128
#   NODE_RANK: 0 for single-node (default)
#
# For remote launch on b200-2:
#   sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@28.89.17.144 \
#     'cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory && \
#      nohup bash scripts/launch_cross_attn_memory.sh arm1_64 > logs/cross_attn_arm1.log 2>&1 &'
#
# For remote launch on b200-3:
#   sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@28.89.17.85 \
#     'cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory && \
#      nohup bash scripts/launch_cross_attn_memory.sh arm2_128 > logs/cross_attn_arm2.log 2>&1 &'

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs
export PYTHONUNBUFFERED=1

ARM=${1:-arm1_64}
NODE_RANK=${2:-0}

SHARD_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
WIKITEXT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy"

case $ARM in
  arm1_64)
    OUTPUT="outputs/cross_attn_memory_arm1_slots64"
    NUM_SLOTS=64
    TOP_K=8
    ;;
  arm2_128)
    OUTPUT="outputs/cross_attn_memory_arm2_slots128"
    NUM_SLOTS=128
    TOP_K=16
    ;;
  *)
    echo "Unknown ARM: $ARM (use arm1_64 | arm2_128)"
    exit 1
    ;;
esac

LOG="logs/cross_attn_memory_${ARM}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "Scheme A: Cross-Attention Memory"
echo "ARM: $ARM (num_slots=$NUM_SLOTS, top_k=$TOP_K)"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo "Node rank: $NODE_RANK"
echo "============================================"

torchrun \
    --nproc_per_node=8 \
    --nnodes=${NNODES:-1} \
    --node_rank=$NODE_RANK \
    --rdzv_id=$$ \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500} \
    scripts/train_cross_attn_memory.py \
    --model ${MODEL} \
    --shard_dir ${SHARD_DIR} \
    --output_dir ${OUTPUT} \
    --wikitext_path ${WIKITEXT} \
    --num_shards 25 --shard_offset 0 \
    --seq_len 4096 --chunks_per_doc 32 \
    --num_slots ${NUM_SLOTS} --top_k ${TOP_K} \
    --use_cross_attn_memory \
    --cross_attn_dropout 0.0 \
    --full_finetune --use_memory \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 5e-6 --min_lr 1e-6 \
    --warmup_steps 200 --max_steps 2000 \
    --eval_interval 100 --save_interval 500 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    2>&1 | tee $LOG
