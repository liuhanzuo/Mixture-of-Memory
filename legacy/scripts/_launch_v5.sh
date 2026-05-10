#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export OUTPUT_DIR=outputs/sparse_memory_ablation_2048_v5
mkdir -p "$OUTPUT_DIR"

exec torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    scripts/train_sparse_memory.py \
    --base_model models/Llama--Llama2-7b \
    --output_dir "$OUTPUT_DIR" \
    --data_path data/pg19_train.jsonl \
    --memory_slots 2048 \
    --top_k 8 \
    --sliding_window 256 \
    --ema_alpha 0.1 \
    --gate_bias_init 2.0 \
    --lr 2e-5 \
    --seq_len 4096 \
    --batch_size 2 \
    --grad_accumulation_steps 4 \
    --max_steps 5000 \
    --log_every 50 \
    --save_every 1000 \
    --warmup_steps 100 \
    --gradient_checkpointing \
    2>&1 | tee "$OUTPUT_DIR/launch.log"
