#!/bin/bash
# Launch full finetune experiments on 3 nodes
# Usage: bash scripts/launch_full_finetune.sh <MODE> [NODE_RANK]
# MODE: full_mem | full_nomem | lora_ref
# NODE_RANK: 0, 1, 2, ... (default 0 for single-node)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

MODE=${1:-full_mem}
SHARD_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
WIKITEXT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy"

# Figure out node rank from hostname or argument
NODE_RANK=${2:-0}

case $MODE in
  full_mem)
    OUTPUT="outputs/full_finetune_mem"
    EXTRA="--full_finetune --use_memory --lr 5e-6"
    ;;
  full_nomem)
    OUTPUT="outputs/full_finetune_nomem"
    EXTRA="--full_finetune --no_memory --lr 5e-6"
    ;;
  lora_ref)
    OUTPUT="outputs/lora_best_ref"
    EXTRA="--lora_finetune --lora_rank 16 --lr 1e-4"
    ;;
  *)
    echo "Unknown mode: $MODE (use full_mem|full_nomem|lora_ref)"
    exit 1
    ;;
esac

mkdir -p logs
LOG="logs/full_finetune_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "Mode: $MODE"
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
    scripts/train_full_finetune.py \
    --shard_dir ${SHARD_DIR} \
    --model ${MODEL} \
    --output_dir ${OUTPUT} \
    --wikitext_path ${WIKITEXT} \
    --num_shards 25 --shard_offset 0 \
    --seq_len 4096 --chunks_per_doc 32 \
    --num_slots 64 --top_k 8 \
    --warmup_steps 200 --max_steps 2000 \
    --eval_interval 100 --save_interval 500 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --kl_weight 0.0 \
    $EXTRA \
    2>&1 | tee $LOG
