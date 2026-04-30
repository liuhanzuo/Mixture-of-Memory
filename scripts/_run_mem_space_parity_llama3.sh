#!/bin/bash
# Memory-Space v0 — parity/ablation smoke (BYPASS memory path).
#
# Wraps every LlamaDecoderLayer with MemorySpaceLayer, then monkey-patches
# each layer's forward to call forward_no_memory (which just calls the
# original wrapped layer).  Expected PPL: ~6-8 on 10 pg19 chunks (Llama-3-8B).
#
# If parity PPL is vanilla-like: the joint-attn forward is the bug source.
# If parity PPL is still inflated: the bug is in wrapping / patching itself.
#
# Usage:
#     PROJECT_DIR=/root/Mixture-of-Memory ./scripts/_run_mem_space_parity_llama3.sh

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/root/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

MODEL="${MODEL:-$PROJECT_DIR/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/mem_space_v0_parity_llama3}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Same knobs as smoke for direct apples-to-apples comparison.
MAX_CHUNKS=10
SEQ_LEN=4096
NUM_SLOTS=512
TOP_K=64
TRAIN_STEPS=0   # eval-only — we want a pure measurement, no optimizer steps
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== mem_space v0 PARITY (single-GPU, bypass_memory) ==="
echo "Model:       $MODEL"
echo "Data:        $DATA"
echo "Output:      $OUTPUT_DIR"
echo "N=$NUM_SLOTS top_k=$TOP_K max_chunks=$MAX_CHUNKS (eval-only)"
echo "CUDA_VISIBLE_DEVICES=$GPU_INDEX"
echo ""

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=1 --master_port=29511 \
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
    --max_train_steps $TRAIN_STEPS \
    --lr 1e-3 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --bypass_memory \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== FINAL eval_results.json (parity) ==="
cat "$OUTPUT_DIR/eval_results.json"
echo ""
PPL=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['ppl'])")
NAN=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['nan_chunks'])")
echo "PARITY PPL=$PPL  NAN_CHUNKS=$NAN"
