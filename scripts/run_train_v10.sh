#!/bin/bash
# RMT++ v10 training launch script
# Usage: bash scripts/run_train_v10.sh

set -euo pipefail

DATA="${DATA:-data/rmt_train_mixed.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rmt_v10}"
BASE_MODEL="${BASE_MODEL:-../models/Llama--Llama2-7b}"
NUM_GPUS="${NUM_GPUS:-8}"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29510 \
    scripts/train_rmt_v10.py \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --num_mem_tokens 16 \
    --segment_length 1024 \
    --max_segments 4 \
    --vary_n_segments \
    --bptt_depth 2 \
    --recon_loss_coef 0.1 \
    --use_importance_routing \
    --num_epochs 3 \
    --lr 2e-5 \
    --rmt_lr 5e-5 \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --warmup_steps 30 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --log_every 10 \
    --save_every 200 \
    --eval_every 100 \
    --ddp
