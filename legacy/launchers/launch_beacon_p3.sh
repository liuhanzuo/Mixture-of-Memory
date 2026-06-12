#!/usr/bin/env bash
# Launch Activation Beacon training on local 8x H20.
# P3: Interleaved compression with dense forced read path.
#
# Base frozen, only beacon projections + embedding trained.
# Random compression ratio per step for multi-ratio generalization.

set -e

PROJECT_ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$PROJECT_ROOT"

export WANDB_API_KEY="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"

PYTHON=${PYTHON_BIN:-.venv/bin/python}
OUTPUT_DIR=outputs/beacon_p3
LOGDIR=logs
mkdir -p "$OUTPUT_DIR" "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/beacon_p3_${TIMESTAMP}.log"

echo "Launching Activation Beacon P3 training..."
echo "Output: $OUTPUT_DIR"
echo "Log: $LOGFILE"

torchrun --nproc_per_node=8 --master_port=29600 \
    scripts/train_beacon.py \
    --model_path models/Meta-Llama-3-8B \
    --output_dir "$OUTPUT_DIR" \
    --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
    --seq_len 8192 \
    --n_beacon 4 \
    --compression_ratios "2,4,8,16,32,64,128" \
    --total_steps 20000 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 4 \
    --batch_size 1 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 2000 \
    --wandb_project mixture-of-memory \
    --wandb_run_name "beacon_p3_n4_seq8k" \
    2>&1 | tee "$LOGFILE"
