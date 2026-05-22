#!/bin/bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./:./third_party/babilong-pkg
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export WANDB_MODE=disabled

/opt/conda/envs/torch-base/bin/python scripts/train_mem_space_fast_mem.py \
    --model_path models/Meta-Llama-3-8B \
    --output_dir outputs/fast_mem_v1_test200 \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --chunk_size 1024 \
    --curriculum "0:1" \
    --total_steps 200 \
    --lr 5e-6 \
    --fast_mem_lr_mult 3.0 \
    --warmup_steps 50 \
    --gradient_accumulation_steps 1 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --selector_temperature 20.0 \
    --slot_value_norm_cap 5.0 \
    --slot_init random \
    --slot_init_noise 0.05 \
    --unfreeze_hidden_to_slot \
    --shared_memory_bank \
    --use_dual_gate \
    --forget_bias_init 2.0 \
    --dual_gate_tanh_new \
    --use_l3_summary \
    --use_fast_mem \
    --fast_mem_heads 4 \
    --fast_mem_d_state 128 \
    --fast_mem_chunk_size 16 \
    --fast_mem_fusion_init -2.0 \
    --gradient_checkpointing \
    --babilong_mix_fraction 0.0 \
    --log_interval 10 \
    --save_interval 1000 \
    --eval_interval 1000 \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --seed 42
