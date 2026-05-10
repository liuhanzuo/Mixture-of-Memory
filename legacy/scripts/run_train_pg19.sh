#!/bin/bash
# RMT PG-19 training launcher
# Usage: bash scripts/run_train_pg19.sh
# Override with: BASE_MODEL=xxx OUTPUT_DIR=xxx bash scripts/run_train_pg19.sh

set -e

PROJECT_DIR="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
BASE_MODEL="${BASE_MODEL:-$PROJECT_DIR/models/Llama--Llama2-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/rmt_pg19}"
SCRIPT="$PROJECT_DIR/scripts/train_rmt_pg19.py"
NUM_GPUS=8
NUM_EPOCHS=5
BATCH_SIZE=1
GRAD_ACCUM=8
LR=5e-5
MAX_SEG=4
SEG_SIZE=1024
MEM_TOKENS=16
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=== RMT PG-19 Training ==="
echo "Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Max segments: $MAX_SEG"
echo "Segment size: $SEG_SIZE"
echo "Memory tokens: $MEM_TOKENS"
echo "LR: $LR"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM accum"
echo ""

nohup torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    $SCRIPT \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --max_n_segments $MAX_SEG \
    --input_size $SEG_SIZE \
    --num_mem_tokens $MEM_TOKENS \
    --curriculum \
    --curriculum_epochs 3 \
    --log_every 10 \
    --save_every 500 \
    --eval_every 200 \
    --eval_samples 4 \
    --ddp \
    > "$LOG_DIR/train_pg19.log" 2>&1 &

echo "Training launched. PID: $!"
echo "Monitor: tail -f $LOG_DIR/train_pg19.log"
echo "GPU: watch -n 5 nvidia-smi"
