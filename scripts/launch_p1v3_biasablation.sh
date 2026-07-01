#!/bin/bash
# P1-v3 inject_gate_bias ablation arm (remote H20-3 / H20-4 on share_304376610).
# Usage: launch_p1v3_biasablation.sh <BIAS> <PORT> <LOGNAME> <SEED>
set -u
BIAS="${1:?inject_gate_bias}"
PORT="${2:?master_port}"
LOGNAME="${3:?log/run name}"
SEED="${4:-42}"

PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_API_KEY="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"
export WANDB_MODE=offline
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg"
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=bond1
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

rm -rf "$PROJECT_ROOT/src/memory/mem_space/__pycache__"
mkdir -p "outputs/$LOGNAME" logs

nohup .venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=8 --master_port="$PORT" \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir "outputs/$LOGNAME" --total_steps 2000 --lr 1e-4 --warmup_steps 100 \
    --chunk_size 1024 --batch_size 1 --num_slots 128 --top_k 16 \
    --selector_dim 128 --selector_temperature 20.0 \
    --load_balance_weight 0.01 --entropy_aux_weight 0.001 \
    --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 \
    --slot_value_norm_cap 5.0 --slot_init strided_token --slot_init_noise 0.0 \
    --writeback_gate_max 1.0 --unfreeze_hidden_to_slot --use_dual_gate \
    --forget_bias_init 2.0 --input_bias_init 0.0 --dual_gate_tanh_new \
    --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
    --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 4 \
    --curriculum "0:2,250:4,1000:8" --bptt_window 2 \
    --no_detach_slots_in_selector --no_slot_delta_clip \
    --inject_gate_bias_init "$BIAS" \
    --save_interval 500 --eval_interval 500 --eval_samples 30 --log_interval 5 \
    --grad_clip 1.0 --proj_grad_clip 0.1 \
    --wandb_project mixture-of-memory --wandb_run_name "$LOGNAME" \
    --dtype bfloat16 --attn_impl sdpa --seed "$SEED" \
    > "logs/$LOGNAME.log" 2>&1 &
echo "PID=$! BIAS=$BIAS PORT=$PORT LOG=logs/$LOGNAME.log"
