#!/bin/bash
# Experiment H6: Middle-Layer Memory + LM2-inspired DUAL-GATE writeback
#
# Purpose: Test whether replacing the H5 full-overwrite slot update with
# LSTM-style content-conditioned gates fixes the cross-chunk regression.
#
# H5 / H5b regression hypothesis: persisting slots across chunks left them at
# the mercy of the next chunk's full overwrite — useful needle slots got
# unconditionally replaced. H6 inserts dual-gate (input + forget) so each
# slot can independently choose to retain old content or admit new.
#
# Reference: LM2 paper (arXiv:2502.06049), src/memory.py:259-263 + create_gates.
#   M_new = g_in * tanh(content) + g_forget * M_prev
#   g_in, g_forget = sigmoid_split(W_n*content + W_m*M_prev + bias)
#   forget_bias=1.0 makes g_forget ≈ sigmoid(1) ≈ 0.73 at init ("remember by default")
#
# Target: Local 8× GPU (b200-1 .143)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h6_dual_gate_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H6: Middle-Layer Memory + DUAL-GATE writeback (LM2-inspired)"
echo "  Write layer: 16"
echo "  Read layers: 18,22,26,30"
echo "  Slots: 64"
echo "  Dual-gate: ON (forget_bias=1.0, input_bias=0.0, tanh on new content)"
echo "Output: outputs/experiment_h6_dual_gate"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h6_dual_gate \
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
