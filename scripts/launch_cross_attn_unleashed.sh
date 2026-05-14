#!/bin/bash
# Experiment A: Unleashed CrossAttn -- remove triple suppression
# Arm 1 (b200-2): --residual_scale 1.0 --cross_attn_lr_factor 100  (remove residual scaling only)
# Arm 2 (b200-3): --residual_scale 1.0 --cross_attn_lr_factor 1    (remove residual scaling + full lr)
#
# Usage:
#   bash scripts/launch_cross_attn_unleashed.sh <ARM> [NODE_RANK]
#   ARM: arm1_noscale | arm2_noscale_fulllr
#   NODE_RANK: 0 for single-node (default)
#
# For remote launch on b200-2:
#   sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@28.89.17.144 \
#     'cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory && \
#      nohup bash scripts/launch_cross_attn_unleashed.sh arm1_noscale > logs/cross_attn_unleashed_arm1.log 2>&1 &'
#
# For remote launch on b200-3:
#   sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@28.89.17.85 \
#     'cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory && \
#      nohup bash scripts/launch_cross_attn_unleashed.sh arm2_noscale_fulllr > logs/cross_attn_unleashed_arm2.log 2>&1 &'

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs
export PYTHONUNBUFFERED=1

ARM=${1:-arm1_noscale}
NODE_RANK=${2:-0}

SHARD_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
WIKITEXT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy"

case $ARM in
  arm1_noscale)
    OUTPUT="outputs/cross_attn_unleashed_arm1"
    RESIDUAL_SCALE=1.0
    LR_FACTOR=100
    ;;
  arm2_noscale_fulllr)
    OUTPUT="outputs/cross_attn_unleashed_arm2"
    RESIDUAL_SCALE=1.0
    LR_FACTOR=1
    ;;
  *)
    echo "Unknown ARM: $ARM (use arm1_noscale | arm2_noscale_fulllr)"
    exit 1
    ;;
esac

LOG="logs/cross_attn_unleashed_${ARM}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "Experiment A: Unleashed CrossAttn"
echo "ARM: $ARM"
echo "residual_scale=$RESIDUAL_SCALE, cross_attn_lr_factor=$LR_FACTOR"
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
    --num_slots 64 --top_k 8 \
    --use_cross_attn_memory \
    --cross_attn_dropout 0.0 \
    --residual_scale ${RESIDUAL_SCALE} \
    --cross_attn_lr_factor ${LR_FACTOR} \
    --full_finetune --use_memory \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 5e-6 --min_lr 1e-6 \
    --warmup_steps 200 --max_steps 2000 \
    --eval_interval 100 --save_interval 500 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    2>&1 | tee $LOG
