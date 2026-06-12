#!/bin/bash
# Experiment H8: H6 with FROZEN base model + low NIAH
#
# Diagnosis (2026-05-10): H7 (low NIAH alone) still got 0% AVG. The real issue
# is that --full_finetune corrupts base 8B's in-context learning even at 500 steps,
# regardless of NIAH supervision strength. Pure dolmino-mix LM training is too
# OOD from BABILong's instruction-following format.
#
# H8 fix: FREEZE the base model entirely. Train ONLY:
#   - cross_attn_modules (write @ L16, read @ L18,22,26,30)
#   - dual_gate_proj_new / dual_gate_proj_mem / dual_gate_bias
#
# This preserves Llama-3-8B's in-context learning ability (which on its own gets
# 12% AVG at 8k) and just teaches the memory module to retrieve relevant info.
#
# Reference: This matches the LoRA / adapter-style approach. Memory module
# parameters are ~tens of millions vs 8B base.
#
# Target: Local 8× GPU (b200-1 once H6 finishes — eta ~30 min)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h8_frozen_base_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H8: FROZEN base + low NIAH (preserve Llama-3-8B in-context learning)"
echo "  Write layer: 16"
echo "  Read layers: 18,22,26,30"
echo "  Slots: 64"
echo "  Dual-gate: ON"
echo "  freeze_base_steps: 99999 (effectively forever - never unfreeze)"
echo "  NIAH lambda: 0.1, mix: 0.10, warmup: 500"
echo "Output: outputs/experiment_h8_frozen_base"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h8_frozen_base \
    --wikitext_path /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
    --niah_data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
    --slot_forward \
    --middle_layer_memory \
    --memory_write_layer 16 \
    --memory_read_layers 18,22,26,30 \
    --memory_init strided \
    --use_dual_gate \
    --forget_bias_init 1.0 \
    --input_bias_init 0.0 \
    --niah_mix_fraction 0.10 \
    --niah_max_N 2 \
    --niah_warmup_steps 500 \
    --lambda_retrieve 0.1 \
    --freeze_base_steps 99999 \
    --num_shards 25 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --num_slots 64 \
    --full_finetune \
    --use_memory \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_steps 200 \
    --max_steps 5000 \
    --eval_interval 200 \
    --save_interval 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    2>&1 | tee $LOGFILE
