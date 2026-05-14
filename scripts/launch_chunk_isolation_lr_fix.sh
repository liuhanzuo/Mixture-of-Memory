#!/bin/bash
# Chunk Isolation — LR Fix Ablation (2026-05-05)
#
# Root cause: cross_attn_lr_factor=1 causes out_proj_norm to grow uncontrollably after warmup.
# Evidence: arm1(sl=256) ratio 0.9852@100 → 1.0126@200; arm2(sl=512) ratio 0.9868@100 → 1.0165@400
# Fix: reduce cross-attn lr by factor 10 or 50 to control out_proj growth.
#
# arm1: cross_attn_lr_factor=10  (effective lr = 5e-7)  -- sl=256, local/b200-1
# arm2: cross_attn_lr_factor=50  (effective lr = 1e-7)  -- sl=256, b200-4 when available
#
# Usage (local):
#   nohup bash scripts/launch_chunk_isolation_lr_fix.sh arm1 > logs/chunk_isolation_lr_fix_arm1.log 2>&1 &
#
# Usage (remote b200-4):
#   sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@28.89.19.134 \
#     'cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory && \
#      nohup bash scripts/launch_chunk_isolation_lr_fix.sh arm2 > logs/chunk_isolation_lr_fix_arm2.log 2>&1 &'

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs
export PYTHONUNBUFFERED=1

ARM=${1:-arm1}

SHARD_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
WIKITEXT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy"

case $ARM in
  arm1)
    LR_FACTOR=10
    SEQ_LEN=256
    OUTPUT="outputs/chunk_isolation_lrfix_arm1_factor10"
    ;;
  arm2)
    LR_FACTOR=50
    SEQ_LEN=256
    OUTPUT="outputs/chunk_isolation_lrfix_arm2_factor50"
    ;;
  *)
    echo "Unknown ARM: $ARM (use arm1 | arm2)"
    exit 1
    ;;
esac

LOG="logs/chunk_isolation_lr_fix_${ARM}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "Chunk Isolation LR Fix — ${ARM}"
echo "cross_attn_lr_factor=${LR_FACTOR} (effective lr=$(python3 -c "print(f'{5e-6/float(${LR_FACTOR}):.1e}')"))"
echo "seq_len=${SEQ_LEN}, chunks_per_doc=128"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo "============================================"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_id=$$ \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:${MASTER_PORT:-29600} \
    scripts/train_cross_attn_memory.py \
    --model ${MODEL} \
    --shard_dir ${SHARD_DIR} \
    --output_dir ${OUTPUT} \
    --wikitext_path ${WIKITEXT} \
    --num_shards 25 \
    --seq_len ${SEQ_LEN} --chunks_per_doc 128 \
    --num_slots 64 --top_k 8 \
    --use_cross_attn_memory \
    --residual_scale 1.0 \
    --cross_attn_lr_factor ${LR_FACTOR} \
    --full_finetune --use_memory \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 5e-6 --min_lr 1e-6 \
    --warmup_steps 200 --max_steps 4000 \
    --eval_interval 100 --save_interval 500 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    2>&1 | tee $LOG
