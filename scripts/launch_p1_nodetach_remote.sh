#!/bin/bash
# P1-v2: Same as P1 but with two critical changes to break gate freeze deadlock:
#   1. --no_detach_slots_in_selector: K_sel gets gradient through routing path
#   2. --no_slot_delta_clip: slot_delta not clipped, stronger inject_gate gradient
#   3. inject_gate bias = -2.0 (g≈0.12 at init, must learn to open)
# Remote H20 (28.59.80.196), shares FS with local. seed=44 for diversity.
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory || exit 1
PYTHON=/opt/conda/envs/torch-base/bin/python

mkdir -p logs outputs/p1_nodetach_remote

nohup $PYTHON -u -m torch.distributed.run --nproc_per_node=8 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir outputs/p1_nodetach_remote \
    --total_steps 2000 --lr 1e-4 --warmup_steps 100 \
    --chunk_size 1024 --batch_size 1 \
    --num_slots 128 --top_k 16 --selector_dim 128 \
    --selector_temperature 20.0 \
    --load_balance_weight 0.01 --entropy_aux_weight 0.001 \
    --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 \
    --slot_value_norm_cap 5.0 \
    --slot_init strided_token --slot_init_noise 0.0 \
    --writeback_gate_max 1.0 --unfreeze_hidden_to_slot \
    --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
    --dual_gate_tanh_new \
    --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
    --shared_memory_bank \
    --gradient_checkpointing --gradient_accumulation_steps 4 \
    --curriculum "0:2,250:4,1000:8" \
    --bptt_window 2 \
    --no_detach_slots_in_selector \
    --no_slot_delta_clip \
    --inject_gate_bias_init -2.0 \
    --save_interval 500 --eval_interval 500 --eval_samples 30 \
    --log_interval 5 --grad_clip 1.0 --proj_grad_clip 0.1 \
    --wandb_project mixture-of-memory \
    --wandb_run_name p1_nodetach_remote \
    --dtype bfloat16 --attn_impl sdpa --seed 44 \
    > logs/p1_nodetach_remote.log 2>&1 &

echo "P1-nodetach training launched, PID=$!"
echo "Log: logs/p1_nodetach_remote.log"
