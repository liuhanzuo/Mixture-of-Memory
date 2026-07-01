#!/bin/bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONPATH=./:./third_party/babilong-pkg
export OMP_NUM_THREADS=4
export WANDB_API_KEY="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"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

/opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir outputs/prepend_all_fastmem_10k_remote_h20 \
    --total_steps 10000 --lr 5e-5 --warmup_steps 500 \
    --chunk_size 1024 --batch_size 1 \
    --num_slots 128 --top_k 128 --selector_dim 128 \
    --selector_temperature 20.0 --writeback_gate_max 0.3 \
    --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
    --key_repulsion_weight 0.0 --slot_value_norm_cap 5.0 \
    --slot_init strided_token --slot_init_noise 0.0 \
    --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
    --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
    --use_fast_mem \
    --gradient_checkpointing --gradient_accumulation_steps 2 \
    --curriculum '0:2,1000:4,3000:8,5000:16' \
    --babilong_mix_fraction 0.0 \
    --save_interval 1000 --eval_interval 500 --log_interval 10 \
    --grad_clip 1.0 --proj_grad_clip 0.5 \
    --wandb_run_name prepend_all_fastmem_10k_remote_h20 \
    --seed 44
