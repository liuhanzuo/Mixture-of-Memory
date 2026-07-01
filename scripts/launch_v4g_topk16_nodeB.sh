#!/bin/bash
# v4g content-routing with PROPER sparsity: top_k=16, num_slots=128
# Fix: previous v4g had top_k=128=num_slots (no selection happening)
# Node B (29.162.241.149), 8x H20

set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

export WANDB_API_KEY="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"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export PYTHONPATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

PYTHON=.venv/bin/python
LOG_FILE=logs/v4g_topk16_nodeB.log
mkdir -p logs outputs/v4g_topk16_nodeB

echo "=== v4g Content Routing topk=16 NodeB ===" | tee $LOG_FILE
echo "Start: $(date)" | tee -a $LOG_FILE

$PYTHON -m torch.distributed.run --nproc_per_node=8 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir outputs/v4g_topk16_nodeB \
    --total_steps 2000 \
    --lr 1e-4 \
    --warmup_steps 100 \
    --chunk_size 1024 \
    --batch_size 1 \
    --num_slots 128 \
    --top_k 16 \
    --selector_dim 128 \
    --selector_temperature 20.0 \
    --load_balance_weight 0.01 \
    --entropy_aux_weight 0.001 \
    --key_repulsion_weight 0.05 \
    --key_repulsion_threshold 0.3 \
    --slot_value_norm_cap 5.0 \
    --slot_init strided_token \
    --slot_init_noise 0.0 \
    --writeback_gate_max 1.0 \
    --unfreeze_hidden_to_slot \
    --use_dual_gate \
    --forget_bias_init 2.0 \
    --input_bias_init 0.0 \
    --dual_gate_tanh_new \
    --use_l3_summary \
    --l3_n_summary 64 \
    --l3_n_layers 2 \
    --l3_n_heads 8 \
    --shared_memory_bank \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --curriculum 0:2,250:4,1000:8 \
    --save_interval 500 \
    --eval_interval 500 \
    --eval_samples 30 \
    --log_interval 5 \
    --grad_clip 1.0 \
    --proj_grad_clip 0.1 \
    --wandb_project mixture-of-memory \
    --wandb_run_name v4g_topk16_nodeB \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --seed 42 \
    2>&1 | tee -a $LOG_FILE
