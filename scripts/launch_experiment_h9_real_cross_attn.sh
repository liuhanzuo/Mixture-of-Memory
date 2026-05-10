#!/bin/bash
# Experiment H9: First REAL cross-attention memory in middle_layer_memory mode
#
# This is the first H-series experiment where cross_attn_modules are actually
# BUILT and USED. H1-H8 all had a bug where --slot_forward caused the
# cross_attn_modules condition to short-circuit to an empty nn.ModuleList().
#
# 3 bugs fixed from H6:
#   Bug #1: cross_attn_modules never constructed (condition excluded slot_forward)
#           -> Now builds CrossAttentionMemoryV2 modules for each read layer
#   Bug #2: read_slots = slots.detach() cut gradient from read layers to slot update
#           -> Now uses cross_attn_modules[i].read() with NO detach
#   Bug #3: _init_slots unconditionally detached slot_values across chunks
#           -> Now preserves gradient graph for truncated BPTT
#
# Architecture:
#   Write layer (L16): joint self-attention with slots, dual-gate writeback
#   Read layers (L18,22,26,30): CrossAttentionMemoryV2 cross-attention
#     Q=hidden_states, K=V=slots from write layer
#     hidden_states += residual_scale * cross_attn_output
#   All other layers: vanilla forward (no memory interaction)
#
# Target: Local 8x GPU (b200-1 .143)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h9_real_cross_attn_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H9: REAL Cross-Attention Memory (first working cross_attn_modules)"
echo "  Write layer: 16 (dual-gate writeback)"
echo "  Read layers: 18,22,26,30 (CrossAttentionMemoryV2 cross-attention)"
echo "  Slots: 64"
echo "  Dual-gate: ON (forget_bias=1.0, input_bias=0.0, tanh on new content)"
echo "  Bug fixes: #1 build cross_attn, #2 remove detach, #3 cross-chunk grad"
echo "Output: outputs/experiment_h9_real_cross_attn"
echo "Log:    $LOGFILE"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h9_real_cross_attn \
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
    --niah_mix_fraction 0.30 \
    --niah_max_N 2 \
    --niah_warmup_steps 100 \
    --lambda_retrieve 1.0 \
    --num_shards 25 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --num_slots 64 \
    --full_finetune \
    --use_memory \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr 5e-6 \
    --min_lr 1e-6 \
    --warmup_steps 200 \
    --max_steps 5000 \
    --eval_interval 200 \
    --save_interval 1000 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    2>&1 | tee $LOGFILE
