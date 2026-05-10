#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=0
export MASTER_PORT=29700

OUTPUT_DIR=outputs/phase1_val_final
mkdir -p "$OUTPUT_DIR"

echo "=== START $(date) ===" > "$OUTPUT_DIR/full.log"
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    scripts/train_sparse_memory.py \
    --base_model models/Llama--Llama2-7b \
    --output_dir "$OUTPUT_DIR" \
    --data_path data/wikitext_tokenized_1024.npy \
    --memory_slots 256 \
    --top_k 8 \
    --sliding_window 128 \
    --ema_alpha 0.1 \
    --lr 2e-5 \
    --seq_len 1024 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --max_steps 200 \
    --log_every 10 \
    --save_every 1000 \
    --warmup_steps 0 \
    --gradient_checkpointing \
    >> "$OUTPUT_DIR/full.log" 2>&1

echo "=== END $(date) EXIT=$? ===" >> "$OUTPUT_DIR/full.log"
