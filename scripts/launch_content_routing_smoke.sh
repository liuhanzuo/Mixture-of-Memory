#!/bin/bash
# 500-step smoke test: content-based routing + TBPTT + recursive L3
# Verifies: top1_sim rises from ~0.016, gates unfreeze, no NaN
set -e
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH=./:./third_party/babilong-pkg
export PYTHONUNBUFFERED=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export WANDB_API_KEY="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"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTHON=${PYTHON_BIN:-.venv/bin/python}
OUTPUT_DIR=${OUTPUT_DIR_OVERRIDE:-outputs/content_routing_smoke}

mkdir -p $OUTPUT_DIR logs

$PYTHON -m torch.distributed.run --nproc_per_node=8 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir $OUTPUT_DIR \
    --total_steps 500 \
    --lr 5e-6 \
    --warmup_steps 50 \
    --chunk_size 1024 \
    --batch_size 1 \
    --num_slots 128 \
    --top_k 128 \
    --selector_dim 128 \
    --selector_temperature 20.0 \
    --load_balance_weight 0.01 \
    --entropy_aux_weight 0.001 \
    --key_repulsion_weight 0.0 \
    --key_repulsion_threshold 0.3 \
    --slot_value_norm_cap 5.0 \
    --slot_init strided_token \
    --slot_init_noise 0.0 \
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
    --curriculum "0:2,250:4" \
    --babilong_mix_fraction 0.15 \
    --babilong_tasks qa1,qa2,qa5 \
    --babilong_lengths 0k,1k,2k,4k \
    --save_interval 250 \
    --eval_interval 0 \
    --eval_samples 30 \
    --log_interval 5 \
    --grad_clip 1.0 \
    --proj_grad_clip 0.1 \
    --wandb_project mixture-of-memory \
    --wandb_run_name content_routing_smoke_500step \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --seed 42
