#!/bin/bash
# Experiment H7: H6 + REDUCED NIAH supervision (fix systemic 0% BABILong failure)
#
# Diagnosis (2026-05-10): All H/H2/H3/H4/H5/H5b/H6/H6b ckpts get 0% BABILong AVG
# because they collapse to "Answer + repeating-token" outputs (e.g., "the take the take",
# "SkipSkipSkip"). Root cause: lambda_retrieve>=1.0 + niah_mix_fraction>=0.30
# made niah_loss dominate gradient signal. niah_loss teacher-forces the model to
# generate needle "12345..." patterns; with mix=30%, this overwhelmed LM loss.
#
# H7 fix: dramatically reduce NIAH supervision strength.
#
# Changes vs H6:
#   - lambda_retrieve: 1.0 → 0.1 (10x reduction)
#   - niah_mix_fraction: 0.30 → 0.10 (3x reduction)
#   - niah_warmup_steps: 100 → 500 (defer NIAH ramp-up to after LM loss is established)
#   - max_steps: 5000 (same)
#
# Target: Local 8× GPU (b200-1 .143)

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h7_low_niah_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H7: H6 + REDUCED NIAH supervision (fix model collapse)"
echo "  Write layer: 16"
echo "  Read layers: 18,22,26,30"
echo "  Slots: 64"
echo "  Dual-gate: ON"
echo "  NIAH lambda: 0.1 (vs H6 1.0)"
echo "  NIAH mix:    0.10 (vs H6 0.30)"
echo "  NIAH warmup: 500 (vs H6 100)"
echo "Output: outputs/experiment_h7_low_niah"
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
    --output_dir /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h7_low_niah \
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
