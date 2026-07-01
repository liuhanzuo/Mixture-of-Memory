#!/bin/bash
# Launch script: FastMem v1 training on remote H20 node (28.59.80.196)
# 8-GPU torchrun with Gated Delta Rule fast-weight memory
#
# Usage:
#   bash scripts/launch_fast_mem_train.sh
#
# Or on the remote node directly:
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   bash scripts/launch_fast_mem_train.sh

set -euo pipefail

# --- Configuration ---
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NPROC_PER_NODE=8
OUTPUT_DIR="${PROJECT_ROOT}/outputs/fast_mem_v1"

# Proxy for wandb
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"

# NCCL settings for H20
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=== FastMem v1 Training Launch ==="
echo "  PROJECT_ROOT: ${PROJECT_ROOT}"
echo "  PYTHON_BIN:   ${PYTHON_BIN}"
echo "  OUTPUT_DIR:   ${OUTPUT_DIR}"
echo "  NPROC:        ${NPROC_PER_NODE}"
echo "  Start time:   $(date)"
echo "=================================="

${PYTHON_BIN} -m torch.distributed.run \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=29501 \
    scripts/train_mem_space_fast_mem.py \
    --model_path "${PROJECT_ROOT}/models/Meta-Llama-3-8B" \
    --output_dir "${OUTPUT_DIR}" \
    --dolmino_path "${PROJECT_ROOT}/MemLong/data/processed/dolmino_0.5B_1024/train" \
    --chunk_size 1024 \
    --curriculum "0:1,5000:2,10000:4,15000:8,25000:16" \
    --total_steps 50000 \
    --lr 5e-6 \
    --fast_mem_lr_mult 3.0 \
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
    --use_fast_mem \
    --fast_mem_heads 4 \
    --fast_mem_d_state 128 \
    --fast_mem_chunk_size 64 \
    --fast_mem_fusion_init -2.0 \
    --gradient_checkpointing \
    --babilong_mix_fraction 0.15 \
    --babilong_tasks "qa1,qa2,qa5" \
    --babilong_lengths "0k,1k,2k,4k" \
    --log_interval 10 \
    --save_interval 5000 \
    --eval_interval 2000 \
    --eval_samples 50 \
    --wandb_project "mixture-of-memory" \
    --wandb_run_name "fast_mem_v1" \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --seed 42 \
    2>&1 | tee "logs/fast_mem_v1_$(date +%Y%m%d_%H%M%S).log"

echo "=== Training finished at $(date) ==="
