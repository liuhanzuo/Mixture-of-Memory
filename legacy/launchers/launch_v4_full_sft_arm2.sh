#!/bin/bash
# Launch v4 full SFT training — ARM 2: lower lr for comparison.
# Arm 1 (b200-2): lr=1e-5
# Arm 2 (b200-3): lr=5e-6
set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

OUTPUT_DIR="outputs/v4_full_sft_lr5e6"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/v4_full_sft_lr5e6_${TIMESTAMP}.log"

mkdir -p logs "${OUTPUT_DIR}"

echo "Launching v4_full_sft (arm2 lr=5e-6) at $(date)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"

torchrun --nproc_per_node=8 scripts/train_v4_full_sft.py \
    --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --output_dir "${OUTPUT_DIR}" \
    --num_slots 4 \
    --top_k 2 \
    --seq_len 4096 \
    --lr 5e-6 \
    --max_steps 2000 \
    --gradient_accumulation_steps 4 \
    --pretrain_ratio 0.9 \
    --pretrain_max_chunks 4500 \
    --memory_max_chunks 500 \
    --chunks_per_doc 32 \
    --eval_interval 100 \
    --skip_chunks 0 \
    2>&1
