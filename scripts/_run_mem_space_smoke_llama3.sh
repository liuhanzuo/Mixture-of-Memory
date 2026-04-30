#!/bin/bash
# Memory-Space v0 — single-GPU smoke driver (10 chunks × 4096, Llama-3-8B).
#
# Runs a 10-step training rollout (selector + gate only — backbone frozen)
# followed by an eval pass.  Outputs:
#     outputs/mem_space_v0_smoke_llama3_fix1/eval_results.json
#
# Gate on caller side: PPL finite, nan_chunks == 0, no traceback.
# Fix1 (2026-04-26): --slot_init random --slot_init_noise 1.0 to avoid the
# oracle-slot-leak pathology diagnosed in
# ops/research_notes/20260426_mem_space_v0_jointattn_diagnosis.md
# (expected PPL in [15, 30] vs pre-fix 406.74).

set -e
set -u

# -------- environment -------- #
PROJECT_DIR="${PROJECT_DIR:-/root/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

# -------- paths -------- #
MODEL="${MODEL:-$PROJECT_DIR/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/mem_space_v0_smoke_llama3_fix1}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# -------- hypers -------- #
MAX_CHUNKS=10
SEQ_LEN=4096
NUM_SLOTS=512
TOP_K=64
TRAIN_STEPS=10
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"

# -------- launch -------- #
echo "=== mem_space v0 SMOKE (single-GPU) ==="
echo "Model:       $MODEL"
echo "Data:        $DATA"
echo "Output:      $OUTPUT_DIR"
echo "N=$NUM_SLOTS top_k=$TOP_K max_chunks=$MAX_CHUNKS train_steps=$TRAIN_STEPS"
echo "CUDA_VISIBLE_DEVICES=$GPU_INDEX"
echo ""

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=1 --master_port=29510 \
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
echo "SMOKE PPL=$PPL  NAN_CHUNKS=$NAN"
