#!/bin/bash
# Launch teacher distillation training for Memory-Space compression.
# Teacher = same Llama-3-8B with memory disabled (full context via sliding window)
# Student = Llama-3-8B with memory enabled (context compressed into slots)
#
# Usage: bash scripts/launch_distill_train.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
MODEL_PATH="${MODEL_PATH:-models/Meta-Llama-3-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/distill_kd}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
PG19_DATA="${PG19_DATA:-data/pg19_chunks_llama3.npy}"
NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
ALPHA_KD="${ALPHA_KD:-0.5}"
ALPHA_LM="${ALPHA_LM:-0.5}"
TEACHER_WINDOW="${TEACHER_WINDOW:-4}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Wandb
export WANDB_API_KEY="${WANDB_API_KEY:-}"

PYTHON="${PYTHON_BIN:-.venv/bin/python}"

echo "=== Teacher Distillation Training ==="
echo "  Model:           $MODEL_PATH"
echo "  Output:          $OUTPUT_DIR"
echo "  GPUs:            $NUM_GPUS"
echo "  Steps:           $TOTAL_STEPS"
echo "  Chunk size:      $CHUNK_SIZE"
echo "  Max seq len:     $MAX_SEQ_LEN"
echo "  KD temperature:  $KD_TEMPERATURE"
echo "  Alpha KD/LM:     $ALPHA_KD / $ALPHA_LM"
echo "  Teacher window:  $TEACHER_WINDOW chunks"
echo "  LR:              $LR"
echo ""

EXTRA_ARGS=""
if [ -n "$INIT_CHECKPOINT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --init_checkpoint $INIT_CHECKPOINT"
fi

torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 \
    scripts/train_mem_space_distill.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --pg19_data "$PG19_DATA" \
    --max_seq_len $MAX_SEQ_LEN \
    --chunk_size $CHUNK_SIZE \
    --batch_size $BATCH_SIZE \
    --total_steps $TOTAL_STEPS \
    --lr $LR \
    --kd_temperature $KD_TEMPERATURE \
    --alpha_kd $ALPHA_KD \
    --alpha_lm $ALPHA_LM \
    --teacher_window_chunks $TEACHER_WINDOW \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --use_dual_gate \
    --forget_bias_init 2.0 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --gradient_checkpointing \
    --log_interval 10 \
    --save_interval 100 \
    $EXTRA_ARGS \
    2>&1 | tee "$OUTPUT_DIR/train.log"
