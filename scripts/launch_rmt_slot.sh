#!/bin/bash
set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONPATH="$PWD:$PYTHONPATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/rmt_slot_medium_${TIMESTAMP}.log"
mkdir -p logs outputs/rmt_slot_medium

echo "=== RMT-Slot Training ==="
echo "Log: $LOGFILE"
echo "Start: $(date)"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train_rmt_slot.py \
    --model_name_or_path /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --output_dir outputs/rmt_slot_medium \
    --num_slots 64 \
    --top_k 8 \
    --segment_length 1024 \
    --max_n_segments 4 \
    --ema_gate_init 0.3 \
    --bptt_depth -1 \
    --lr 5e-6 \
    --warmup_steps 200 \
    --max_steps 2000 \
    --gradient_accumulation_steps 4 \
    --eval_interval 200 \
    --save_interval 1000 \
    2>&1 | tee "$LOGFILE"

echo "=== Done: $(date) ==="
