#!/bin/bash
# Train SparseMemoryLlamaForCausalLM with SELECTIVE memory writing
# Importance-based top-K writing: only top-K important tokens update memory
# Target: ~99% content retention vs 3.4% with legacy all-token writing
#
# Usage:
#   bash scripts/run_train_selective_write.sh [NUM_GPUS]
#
# Recommended: write_top_k=8 (matches retrieval top_k for balanced read/write)
# For aggressive compression: write_top_k=4
# For safety net (legacy compat): write_top_k=0

set -euo pipefail

NUM_GPUS=${1:-8}
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

BASE_MODEL="${BASE_MODEL:-../models/Qwen--Qwen3-8b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sparse_selective_write}"
DATA_PATH="${DATA_PATH:-data/rmt_train_mixed.jsonl}"

# Memory config
MEMORY_SLOTS=${MEMORY_SLOTS:-128}
TOP_K=${TOP_K:-8}              # retrieval top-K
WRITE_TOP_K=${WRITE_TOP_K:-8}  # write top-K (SELECTIVE writing)
SLIDING_WINDOW=${SLIDING_WINDOW:-256}
EMA_ALPHA=${EMA_ALPHA:-0.1}
IMPORTANCE_MODE=${IMPORTANCE_MODE:-combined}  # magnitude | attention_surprise | combined

# Training config
SEQ_LEN=${SEQ_LEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-2e-5}
MAX_STEPS=${MAX_STEPS:-5000}
SAVE_EVERY=${SAVE_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-50}

# Select GPUs
if [ "$NUM_GPUS" -eq 4 ]; then
    GPU_IDS="${GPU_IDS:-0,1,2,3}"
elif [ "$NUM_GPUS" -eq 8 ]; then
    GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
else
    GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
fi

echo "=== Sparse Memory Training with SELECTIVE Writing ==="
echo "GPUs: $NUM_GPUS ($GPU_IDS)"
echo "Memory: slots=$MEMORY_SLOTS, top_k=$TOP_K, write_top_k=$WRITE_TOP_K, window=$SLIDING_WINDOW"
echo "Selective write: mode=$IMPORTANCE_MODE, ema_alpha=$EMA_ALPHA"
echo "Expected retention: ~$(python3 -c "print(f'{(1-$EMA_ALPHA)**($WRITE_TOP_K/$MEMORY_SLOTS)*100:.1f}')")%"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS \
    scripts/train_sparse_memory.py \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --memory_slots $MEMORY_SLOTS \
    --top_k $TOP_K \
    --write_top_k $WRITE_TOP_K \
    --importance_mode $IMPORTANCE_MODE \
    --sliding_window $SLIDING_WINDOW \
    --ema_alpha $EMA_ALPHA \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --grad_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --max_steps $MAX_STEPS \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --warmup_steps 100 \
    --gradient_checkpointing
