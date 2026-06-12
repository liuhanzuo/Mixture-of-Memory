#!/bin/bash
# Experiment H6b: Middle-Layer Memory + DUAL-GATE writeback + NIAH-aggressive
#
# Variant of H6: same dual-gate writeback architecture, BUT with the
# H5b-style aggressive NIAH retrieval signal:
#   - niah_mix_fraction=0.50 (vs H6's 0.30)
#   - lambda_retrieve=2.0    (vs H6's 1.0)
#
# This ablates whether the dual-gate alone is enough (H6) or whether the
# stronger retrieval supervision (H5b's setting) is also needed for the gate
# to learn meaningful per-slot forget patterns.
#
# Reference: LM2 paper (arXiv:2502.06049).
#
# Target: Remote 8× GPU (b200-3 .85)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h6b_dual_gate_aggressive_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H6b: Dual-Gate + NIAH-aggressive (LM2-inspired)"
echo "  Write layer: 16"
echo "  Read layers: 18,22,26,30"
echo "  Slots: 64"
echo "  Dual-gate: ON (forget_bias=1.0)"
echo "  NIAH mix: 0.50, lambda_retrieve: 2.0"
echo "Output: outputs/experiment_h6b_dual_gate_aggressive"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h6b_dual_gate_aggressive \
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
    --niah_mix_fraction 0.50 \
    --niah_max_N 2 \
    --niah_warmup_steps 100 \
    --lambda_retrieve 2.0 \
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
