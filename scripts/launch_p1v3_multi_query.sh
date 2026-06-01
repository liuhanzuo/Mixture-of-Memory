#!/bin/bash
# P1-v3 with multi_query routing (2026-06-01)
# Root cause (updated): single global chunk query (max-pool OR mean-pool) cannot
# preserve intra-chunk heterogeneity -> routing collapses to uniform.
# Fix: use L3's 64 summary tokens as 64 independent sub-queries, aggregate per-slot
# relevance via logsumexp over queries, then global top-k.
#
# Usage:
#   LOCAL:  bash scripts/launch_p1v3_multi_query.sh
#   REMOTE: PYTHON_BIN=/opt/conda/envs/torch-base/bin/python WANDB_MODE=offline bash scripts/launch_p1v3_multi_query.sh
#
# 300-step health check: set TOTAL_STEPS=300 to fail fast if routing doesn't recover.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PORT="${PORT:-29730}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-p1v3_multi_query}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
MULTI_QUERY_TAU="${MULTI_QUERY_TAU:-1.0}"
ROUTING_POOL_MODE="${ROUTING_POOL_MODE:-multi_query}"
SELECTOR_TEMPERATURE="${SELECTOR_TEMPERATURE:-20.0}"
L3_DIVERSITY_WEIGHT="${L3_DIVERSITY_WEIGHT:-0.1}"
L3_DIVERSITY_THRESHOLD="${L3_DIVERSITY_THRESHOLD:-0.5}"
LOG="${PROJECT_ROOT}/logs/${RUN_NAME}.log"

cd "$PROJECT_ROOT"

# Clear pycache to pick up code changes
rm -rf src/memory/mem_space/__pycache__

export WANDB_API_KEY="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"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

nohup $PYTHON_BIN -u -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=$PORT \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --output_dir outputs/${RUN_NAME} \
    --total_steps $TOTAL_STEPS \
    --lr 1e-4 \
    --warmup_steps 100 \
    --chunk_size 1024 \
    --batch_size 1 \
    --num_slots 128 \
    --top_k 16 \
    --selector_dim 128 \
    --selector_temperature $SELECTOR_TEMPERATURE \
    --load_balance_weight 0.01 \
    --entropy_aux_weight 0.001 \
    --key_repulsion_weight 1.0 \
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
    --curriculum "0:2,250:4,1000:8" \
    --bptt_window 2 \
    --no_detach_slots_in_selector \
    --no_slot_delta_clip \
    --inject_gate_bias_init -2.0 \
    --routing_pool_mode $ROUTING_POOL_MODE \
    --multi_query_tau $MULTI_QUERY_TAU \
    --l3_diversity_weight $L3_DIVERSITY_WEIGHT \
    --l3_diversity_threshold $L3_DIVERSITY_THRESHOLD \
    --save_interval 500 \
    --eval_interval ${EVAL_INTERVAL:-0} \
    --eval_samples 30 \
    --log_interval 5 \
    --grad_clip 1.0 \
    --proj_grad_clip 0.1 \
    --wandb_project mixture-of-memory \
    --wandb_run_name ${RUN_NAME} \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --seed $SEED \
    > "$LOG" 2>&1 &

echo "PID=$!"
echo "Log: $LOG"
