#!/bin/bash
# Experiment G: Pure baseline -- continued pretraining without memory slots
# Purpose: isolate backbone adaptation contribution from slot mechanism contribution
# Target: b200-1 (local), 8 GPU
#
# This is the control group for approach C (launch_approach_c_strided.sh).
# Same data, same hyperparameters, but --no_memory: no cross-attention, no slots.
# The only difference is the absence of any memory mechanism.
#
# Comparison matrix:
#   Experiment C (slot_forward, memory_init=strided): slot mechanism + backbone adaptation
#   Experiment G (this script, --no_memory):          backbone adaptation only
#   Delta (C - G):                                    slot mechanism contribution
set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_g_baseline_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment G: Pure Baseline (no memory)"
echo "Output: outputs/experiment_g_baseline"
echo "Log: $LOGFILE"
echo "============================================"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train_cross_attn_memory.py \
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --shard_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3 \
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_g_baseline \
    --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
    --no_memory \
    --num_shards 25 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 5e-6 \
    --min_lr 1e-6 \
    --warmup_steps 200 \
    --max_steps 5000 \
    --eval_interval 200 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    2>&1 | tee $LOGFILE
