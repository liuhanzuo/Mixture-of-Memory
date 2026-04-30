#!/bin/bash
# Memory-Space v0 — mem_path ON but EVAL-ONLY (no optimizer steps).
#
# Isolates "the forward path is intrinsically broken" from "10 training steps
# destroy the selector".  Same knobs as smoke except train_steps=0.

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/root/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

MODEL="${MODEL:-$PROJECT_DIR/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/mem_space_v0_evalonly_llama3}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

MAX_CHUNKS=10
SEQ_LEN=4096
NUM_SLOTS=512
TOP_K=64
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== mem_space v0 EVAL-ONLY (mem path ON, 0 train steps) ==="
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=1 --master_port=29512 \
    "$PROJECT_DIR/scripts/train_mem_space_pg19.py" \
    --model "$MODEL" \
    --data "$DATA" \
    --max_chunks $MAX_CHUNKS \
    --skip_chunks 1000 \
    --seq_len $SEQ_LEN \
    --batch_size 1 \
    --num_slots $NUM_SLOTS \
    --top_k $TOP_K \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --writeback_warmup_steps 0 \
    --load_balance_weight 0.01 \
    --max_train_steps 0 \
    --lr 1e-3 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --output_dir "$OUTPUT_DIR"

echo ""
cat "$OUTPUT_DIR/eval_results.json"
echo ""
PPL=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['ppl'])")
NAN=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['nan_chunks'])")
echo "EVAL-ONLY PPL=$PPL NAN=$NAN"
