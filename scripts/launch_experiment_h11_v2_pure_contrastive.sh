#!/bin/bash
# Experiment H11_v2: Pure Contrastive — REAL this time
#
# H11_v1 (killed 2026-05-10 12:35): contrastive_weight=5.0 was a silent no-op
# because forward_niah_sample skipped the contrastive path under --slot_forward
# (3 guards at L260/L281/L302). Combined with lambda_retrieve=0.01, H11_v1 was
# essentially pure LM finetune — useless for testing the contrastive hypothesis.
#
# H11_v2 (commit 461d78c): contrastive path now actually fires under
# slot_forward + middle_layer_memory. _forward_middle_layer_memory captures
# attn_weights/logits at each read layer, forward_niah_sample uses them for
# InfoNCE.
#
# Config changes vs H11_v1:
#   - lambda_retrieve: 0.01 → 0.1 (raise — contrastive will now actually run,
#     so we want some LM signal too, but contrastive is dominant)
#   - contrastive_weight: 5.0 (unchanged — strong supervision)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h11_v2_pure_contrastive_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H11_v2: Pure Contrastive (commit 461d78c contrastive fix)"
echo "  NIAH: lambda=0.1 (low LM), contrastive=5.0 (strong)"
echo "  Hypothesis: InfoNCE on cross-attn read weights can teach retrieval"
echo "Output: outputs/experiment_h11_v2_pure_contrastive"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h11_v2_pure_contrastive \
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
    --contrastive_weight 5.0 \
    --contrastive_temperature 0.1 \
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
    --save_interval 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    2>&1 | tee $LOGFILE
