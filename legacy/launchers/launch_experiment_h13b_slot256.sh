#!/bin/bash
# Experiment H13b: num_slots=256 ablation (vs H11_v2's 64, H13's 128)
#
# Hypothesis: 4x more memory capacity. Pairs with H13 to test the
# capacity scaling curve (64 -> 128 -> 256). If H13 helps but H13b
# doesn't, capacity saturates around 128. If H13b > H13 > H11_v2,
# more is still better.
#
# Identical to H11_v2 except --num_slots 256.
# Cluster target: b200-6 (28.89.18.190)

set -e
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/experiment_h13b_slot256_$(date +%Y%m%d_%H%M).log

echo "============================================"
echo "Experiment H13b: num_slots=256 (vs H11_v2 64, H13 128)"
echo "  contrastive=5.0, lambda_retrieve=0.1, num_slots=256"
echo "Output: outputs/experiment_h13b_slot256"
echo "Log:    $LOGFILE"
echo "============================================"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train_cross_attn_memory.py \
    --model /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Llama--Llama3-8b \
    --shard_dir /apdcephfs_wzc1/share_304376610/pighzliu_code/data/dolmino-mix-1124-llama3 \
    --output_dir /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/experiment_h13b_slot256 \
    --wikitext_path /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy \
    --niah_data /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
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
    --num_slots 256 \
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
