#!/bin/bash
# Experiment H9b: H9 with conservative NIAH (lower lambda + mix)
#
# H9 (b200-1) uses lambda_retrieve=1.0, niah_mix=0.30 — same as H6 default.
# H9b tests whether reduced NIAH supervision helps preserve in-context learning
# on top of the architecture fix.
#
# Same code commit (746cd93), same architecture, only NIAH knobs differ:
#   - lambda_retrieve: 1.0 → 0.1 (10x reduction)
#   - niah_mix_fraction: 0.30 → 0.10 (3x reduction)
#   - niah_warmup_steps: 100 → 500 (defer NIAH to after LM stabilizes)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h9b_low_niah_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H9b: REAL Cross-Attention + LOW NIAH supervision"
echo "  Write layer: 16 (dual-gate)"
echo "  Read layers: 18,22,26,30 (CrossAttentionMemoryV2)"
echo "  Slots: 64"
echo "  FROZEN base: 235M trainable / 8B"
echo "  NIAH lambda: 0.1 (vs H9 1.0)"
echo "  NIAH mix:    0.10 (vs H9 0.30)"
echo "  NIAH warmup: 500 (vs H9 100)"
echo "Output: outputs/experiment_h9b_low_niah"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h9b_low_niah \
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
    --num_shards 25 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --num_slots 64 \
    --full_finetune \
    --use_memory \
    --freeze_base_steps 99999 \
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
