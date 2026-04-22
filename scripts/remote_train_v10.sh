#!/bin/bash
# RMT++ v10 remote training launcher for L20A nodes
# Usage: bash scripts/remote_train_v10.sh --config <l0|l0l1|l0l2|l0l1l2> [--output_dir <dir>]

set -euo pipefail

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

PYTHON="/opt/conda/envs/torch-base/bin/python3"
TORCHRUN="/opt/conda/envs/torch-base/bin/torchrun"

DATA="${DATA:-data/rmt_train_mixed.jsonl}"
BASE_MODEL="${BASE_MODEL:-/root/Mixture-of-Memory/models/Llama--Llama2-7b}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29510}"

# Parse args
CONFIG=""
EXTRA_OUTPUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --output_dir) EXTRA_OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 --config <l0|l0l1|l0l2|l0l1l2>"
    exit 1
fi

# Config-specific output dir
case "$CONFIG" in
    l0)      OUTPUT_DIR="${EXTRA_OUTPUT:-outputs/rmt_v10_l0}" ;;
    l0l1)    OUTPUT_DIR="${EXTRA_OUTPUT:-outputs/rmt_v10_l0l1}" ;;
    l0l2)    OUTPUT_DIR="${EXTRA_OUTPUT:-outputs/rmt_v10_l0l2}" ;;
    l0l1l2)  OUTPUT_DIR="${EXTRA_OUTPUT:-outputs/rmt_v10_l0l1l2}" ;;
    *)       echo "Unknown config: $CONFIG"; exit 1 ;;
esac

# Build extra flags for L1/L2
L1L2_FLAGS=""
case "$CONFIG" in
    l0)      ;; # baseline, no extra flags
    l0l1)    L1L2_FLAGS="--use_l1" ;;
    l0l2)    L1L2_FLAGS="--use_l2" ;;
    l0l1l2)  L1L2_FLAGS="--use_l1 --use_l2" ;;
esac

echo "========================================"
echo "RMT v10 Remote Training"
echo "Config:       $CONFIG"
echo "Output:       $OUTPUT_DIR"
echo "GPUs:         $NUM_GPUS"
echo "Master port:  $MASTER_PORT"
echo "Extra flags:  $L1L2_FLAGS"
echo "========================================"

cd /root/Mixture-of-Memory/

$TORCHRUN \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train_rmt_v10.py \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --num_mem_tokens 16 \
    --segment_length 1024 \
    --max_segments 4 \
    --vary_n_segments \
    --bptt_depth 2 \
    --recon_loss_coef 0.1 \
    --use_importance_routing \
    --full_finetune \
    --num_epochs 5 \
    --lr 1e-5 \
    --rmt_lr 1e-4 \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --warmup_steps 30 \
    --log_every 10 \
    --save_every 100 \
    --eval_every 100 \
    --seed 42 \
    $L1L2_FLAGS \
    --ddp
