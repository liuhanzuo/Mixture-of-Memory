#!/bin/bash
# Memory-Space v0 — 8-GPU full eval driver (200 chunks × 4096, Llama-3-8B).
#
# Same driver / config as the smoke script but scaled: 8 GPUs via torchrun,
# 200 chunks.  Outputs:
#     outputs/mem_space_v0_full_llama3_fix1_fix2/eval_results.json
#
# Fix1 (2026-04-26): --slot_init random --slot_init_noise 1.0 (avoids oracle-
# slot-leak pathology diagnosed in
# ops/research_notes/20260426_mem_space_v0_jointattn_diagnosis.md). Fix2 is
# architectural and lives in src/memory/mem_space/layer.py (slot-streaming mask
# guard).  Combined smoke PPL=71.92 < 100 red-line.
#
# TRAIN_STEPS=200 so the selector + gate + load-balance aux get a longer
# training rollout than the 10-step smoke; this addresses the residual-gap
# caveat (aux loss not yet converged at 10 steps).

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/root/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

MODEL="${MODEL:-$PROJECT_DIR/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/mem_space_v0_full_llama3_fix1_fix2}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

MAX_CHUNKS=200
SEQ_LEN=4096
NUM_SLOTS=512
TOP_K=64
TRAIN_STEPS=200
NUM_GPUS=8

echo "=== mem_space v0 FULL (8-GPU, fix1+fix2) ==="
echo "Model:   $MODEL"
echo "Data:    $DATA"
echo "Output:  $OUTPUT_DIR"
echo "N=$NUM_SLOTS top_k=$TOP_K max_chunks=$MAX_CHUNKS train_steps=$TRAIN_STEPS"
echo ""

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$NUM_GPUS --master_port=29511 \
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
    --slot_init random \
    --slot_init_noise 1.0 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== FINAL eval_results.json ==="
cat "$OUTPUT_DIR/eval_results.json"
echo ""
PPL=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['ppl'])")
NAN=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['nan_chunks'])")
echo "FULL PPL=$PPL  NAN_CHUNKS=$NAN"
