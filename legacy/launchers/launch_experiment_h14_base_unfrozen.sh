#!/bin/bash
# Experiment H14: unfreeze base 8B after slot warmup (vs H9's permanent freeze)
#
# H9 default: --freeze_base_steps 99999 (base model frozen forever).
# H14 hypothesis: after slots have warmed up (step >= 1000), allowing the
# base 8B params to learn alongside memory may produce stronger retrieval
# integration, since gradient flows into base attention/MLP and not just
# the memory-side parameters.
#
# Cluster target: h20-4 (28.58.246.254). H20 has 97.8GB VRAM, half of B200,
# so seq_len reduced from 4096 -> 2048 and chunks_per_doc doubled 32 -> 64
# to keep ~131k tokens/doc (matches H9-H13).

set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h14_base_unfrozen_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H14: unfreeze base after step 1000 (vs H9 permanent freeze)"
echo "  contrastive=5.0, lambda_retrieve=0.1, num_slots=64"
echo "  freeze_base_steps=1000, seq_len=2048, chunks_per_doc=64"
echo "Output: outputs/experiment_h14_base_unfrozen"
echo "Log:    $LOGFILE"
echo "============================================"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train_cross_attn_memory.py \
    --model /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Llama--Llama3-8b \
    --shard_dir /apdcephfs_zwfy6/share_304376610/pighzliu_code/data/dolmino-mix-1124-llama3 \
    --output_dir /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/experiment_h14_base_unfrozen \
    --wikitext_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
    --niah_data /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
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
    --freeze_base_steps 1000 \
    --num_shards 25 \
    --seq_len 2048 \
    --chunks_per_doc 64 \
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
