#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export OUTPUT_DIR=outputs/sparse_memory_ablation_512_v5
export MEMORY_SLOTS=512
mkdir -p "$OUTPUT_DIR"

# Run with 8 GPUs, smaller memory config that should definitely fit
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    scripts/train_sparse_memory.py \
    --base_model models/Llama--Llama2-7b \
    --output_dir "$OUTPUT_DIR" \
    --data_path data/pg19_train.jsonl \
    --memory_slots $MEMORY_SLOTS \
    --top_k 8 \
    --sliding_window 256 \
    --ema_alpha 0.1 \
    --gate_bias_init 2.0 \
    --lr 2e-5 \
    --seq_len 2048 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --max_steps 50 \
    --log_every 5 \
    --save_every 1000 \
    --warmup_steps 0 \
    --gradient_checkpointing \
    > "$OUTPUT_DIR/stdout.log" 2> "$OUTPUT_DIR/stderr.log"