#!/bin/bash
# Original RMT training launcher for L20A nodes
# Minimal reproduction using Bulatov's MemoryCell + RecurrentWrapper
# Full finetuning, no LoRA

set -euo pipefail

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

PYTHON="/opt/conda/envs/torch-base/bin/python3"
TORCHRUN="/opt/conda/envs/torch-base/bin/torchrun"

DATA="${DATA:-data/rmt_train_mixed.jsonl}"
BASE_MODEL="${BASE_MODEL:-/root/Mixture-of-Memory/models/Llama--Llama2-7b}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29510}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rmt_original}"

echo "========================================"
echo "Original RMT Training"
echo "Output:       $OUTPUT_DIR"
echo "GPUs:         $NUM_GPUS"
echo "Base model:   $BASE_MODEL"
echo "Data:         $DATA"
echo "========================================"

cd /root/Mixture-of-Memory/

$TORCHRUN \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train_rmt_original.py \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --num_mem_tokens 16 \
    --input_size 1024 \
    --max_n_segments 4 \
    --vary_n_segments \
    --bptt_depth -1 \
    --segment_alignment right \
    --num_epochs 3 \
    --lr 5e-5 \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --log_every 10 \
    --save_every 100 \
    --eval_every 50 \
    --eval_samples 4 \
    --seed 42 \
    --ddp
