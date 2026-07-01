#!/bin/bash
# Test: Run Dolmino CPT (no FastMem) on remote H20 to verify if NVLink issue
# is specific to FLA Triton kernel or general hardware failure.
# If this survives >3h → problem is FLA-specific, not general NVLink damage.

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONUNBUFFERED=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export WANDB_MODE=disabled
export PYTHONPATH="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory:/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg"
export TOKENIZERS_PARALLELISM=false

# Same NCCL config as local H20 (NO special workarounds — testing vanilla)
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p outputs/dolmino_cpt_remote_test logs

/opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29501 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --output_dir outputs/dolmino_cpt_remote_test \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --chunk_size 1024 \
    --curriculum "0:1,5000:2,10000:4,15000:8,25000:16" \
    --total_steps 50000 \
    --lr 5e-6 \
    --warmup_steps 1000 \
    --gradient_accumulation_steps 4 \
    --grad_clip 1.0 \
    --proj_grad_clip 0.1 \
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
    --input_bias_init 0.0 \
    --dual_gate_tanh_new \
    --use_l3_summary \
    --l3_n_summary 64 \
    --l3_n_layers 2 \
    --l3_n_heads 8 \
    --gradient_checkpointing \
    --babilong_mix_fraction 0.15 \
    --babilong_tasks qa1,qa2,qa5 \
    --babilong_lengths 0k,1k,2k,4k \
    --log_interval 10 \
    --save_interval 5000 \
    --eval_interval 5000 \
    --eval_samples 20 \
    --wandb_project "" \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --seed 42
