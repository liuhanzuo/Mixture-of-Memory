#!/bin/bash
# Phase 1 validation: 200 steps, check gradient flow via loss and fusion_grad_norm
# gate_bias_init is NOT an arg for train_sparse_memory.py; alpha init is controlled by model config defaults (init_alpha=0.1, max_alpha=0.5)
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=0
export MASTER_PORT=29600
OUTPUT_DIR=outputs/phase1_val_test
mkdir -p "$OUTPUT_DIR"

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    scripts/train_sparse_memory.py \
    --base_model models/Llama--Llama2-7b \
    --output_dir "$OUTPUT_DIR" \
    --data_path data/pg19_train.jsonl \
    --memory_slots 1024 \
    --top_k 8 \
    --sliding_window 256 \
    --ema_alpha 0.1 \
    --lr 2e-5 \
    --seq_len 4096 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --max_steps 200 \
    --log_every 10 \
    --save_every 1000 \
    --warmup_steps 0 \
    --gradient_checkpointing \
    2>&1 | tee "$OUTPUT_DIR/full.log"

echo "EXIT_CODE=$?"
