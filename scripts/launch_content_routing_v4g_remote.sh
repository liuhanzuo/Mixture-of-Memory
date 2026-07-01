#!/bin/bash
# Content Routing v4g — Remote H20
# Key change: writeback_gate_max=1.0 (was 0.3) to fix frozen gates
# Confirmed from v4f: key_repulsion=0.05 stabilizes routing (key_max_cos=0.57-0.71)
# Problem: gates frozen at beta=0.15 because writeback_gate_max=0.3 caps gradient at 0.075
# Fix: writeback_gate_max=1.0 → initial beta=0.5, gradient=0.25 (3.3x stronger)

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH=./:./third_party/babilong-pkg
export PYTHONUNBUFFERED=1

# Wandb
export WANDB_API_KEY="${WANDB_API_KEY}"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

PYTHON=${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}
LOG_FILE=logs/content_routing_v4g_remote.log

echo "=== Content Routing v4g Remote (writeback_gate_max=1.0) ===" | tee $LOG_FILE
echo "Start: $(date)" | tee -a $LOG_FILE

$PYTHON -m torch.distributed.run --nproc_per_node=8 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir outputs/content_routing_v4g_remote \
    --total_steps 2000 \
    --lr 1e-4 \
    --warmup_steps 100 \
    --chunk_size 1024 \
    --batch_size 1 \
    --num_slots 128 \
    --top_k 128 \
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
    --babilong_mix_fraction 0.15 \
    --babilong_tasks qa1,qa2,qa5 \
    --babilong_lengths 0k,1k,2k,4k \
    --save_interval 500 \
    --eval_interval 0 \
    --eval_samples 30 \
    --log_interval 5 \
    --grad_clip 1.0 \
    --proj_grad_clip 0.1 \
    --wandb_project mixture-of-memory \
    --wandb_run_name content_routing_v4g_remote \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --seed 44 \
    2>&1 | tee -a $LOG_FILE
