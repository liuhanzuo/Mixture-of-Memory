#!/bin/bash
# P1 experiment: windowed BPTT (bptt_window=2) + content-conditioned inject gate
# Goal: fix gate freeze + LongBench generalization (baseline F1=34, v4g_topk16_v2 F1=12.66)
# Local H20, 8 GPUs, seed=42. Based on launch_v4g_topk16_v2_remote.sh + commit 86ab8da.
export WANDB_API_KEY="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"

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory || exit 1
PYTHON=.venv/bin/python

mkdir -p logs outputs/p1_bptt_contentgate_local

nohup $PYTHON -u -m torch.distributed.run --nproc_per_node=8 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir outputs/p1_bptt_contentgate_local \
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
    --save_interval 500 --eval_interval 500 --eval_samples 30 \
    --log_interval 5 --grad_clip 1.0 --proj_grad_clip 0.1 \
    --wandb_project mixture-of-memory \
    --wandb_run_name p1_bptt_contentgate_local \
    --dtype bfloat16 --attn_impl sdpa --seed 42 \
    > logs/p1_bptt_contentgate_local.log 2>&1 &

echo "P1 training launched, PID=$!"
echo "Log: logs/p1_bptt_contentgate_local.log"
