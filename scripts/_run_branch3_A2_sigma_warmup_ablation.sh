#!/bin/bash
# Run 2/3: Branch-3 A.2 σ × warmup 12-cell ablation, sharded across 3 B200 nodes.
#
# Purpose: tighten the σ × warmup interaction around the winner config
# (σ=0.02, warmup=500, PPL=1.9051) to see whether nearby cells do better.
#
# Grid (12 cells): σ ∈ {0.01, 0.02, 0.05, 0.1} × warmup ∈ {200, 500, 1000}.
# Shard: NODE_IDX=0 → σ=0.01 row (4 cells), 1 → σ=0.02 row, 2 → σ=0.05 row.
# σ=0.1 row is deferred (node budget = 3).  (σ=0.02 cell 2 reproduces 1.9051
# as a within-node sanity check.)
#
# Usage (launched separately per node):
#     NODE_IDX=0 bash scripts/_run_branch3_A2_sigma_warmup_ablation.sh  # b200-2
#     NODE_IDX=1 bash scripts/_run_branch3_A2_sigma_warmup_ablation.sh  # b200-3
#     NODE_IDX=2 bash scripts/_run_branch3_A2_sigma_warmup_ablation.sh  # b200-4
#
# Per cell: 8×B200, 200 steps × 4096 tokens × 200 chunks, ≈6 min wall.
# Per node: 3 × ≈6 min = ≈18 min wall (3 cells, warmup ∈ {200,500,1000}).

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

NODE_IDX="${NODE_IDX:-0}"

# σ row per node
case "$NODE_IDX" in
  0) SIGMA=0.01 ;;
  1) SIGMA=0.02 ;;
  2) SIGMA=0.05 ;;
  *) echo "ERROR: NODE_IDX must be 0,1,2 (got $NODE_IDX)"; exit 2 ;;
esac

MODEL="${MODEL:-$PROJECT_DIR/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MAX_CHUNKS=200
SEQ_LEN=4096
NUM_SLOTS=512
TOP_K=64
TRAIN_STEPS=200
NUM_GPUS=8

export TOKENIZERS_PARALLELISM=false

run_cell() {
    local warmup=$1
    local tag="sigma${SIGMA}_warmup${warmup}"
    local OUTPUT_DIR="$PROJECT_DIR/outputs/branch3_A2_ablation_${tag}"
    mkdir -p "$OUTPUT_DIR"

    # pick a per-cell master_port so 3 runs on same node don't collide (they
    # run serially here, but sharing a port across ranks after a failure is
    # unreliable). offset: NODE_IDX*10 + warmup_slot.
    local warmup_slot=0
    case "$warmup" in
      200)  warmup_slot=1 ;;
      500)  warmup_slot=2 ;;
      1000) warmup_slot=3 ;;
    esac
    local PORT=$((29530 + NODE_IDX * 10 + warmup_slot))

    echo ""
    echo "=== ABLATION CELL: σ=${SIGMA} warmup=${warmup} (port=${PORT}) ==="

    torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT \
        "$PROJECT_DIR/scripts/train_mem_space_pg19.py" \
        --model "$MODEL" \
        --data "$DATA" \
        --max_chunks $MAX_CHUNKS \
        --skip_chunks 200 \
        --seq_len $SEQ_LEN \
        --batch_size 1 \
        --num_slots $NUM_SLOTS \
        --top_k $TOP_K \
        --selector_dim 128 \
        --writeback_gate_max 0.3 \
        --writeback_warmup_steps $warmup \
        --load_balance_weight 0.01 \
        --max_train_steps $TRAIN_STEPS \
        --lr 1e-3 \
        --attn_impl sdpa \
        --dtype bfloat16 \
        --slot_init random \
        --slot_init_noise $SIGMA \
        --shared_memory_bank \
        --unfreeze_hidden_to_slot \
        --output_dir "$OUTPUT_DIR"

    echo "=== CELL DONE: $tag ==="
    cat "$OUTPUT_DIR/eval_results.json" || echo "(missing eval_results)"
    local PPL=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['ppl'])")
    local NAN=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['nan_chunks'])")
    echo "ABLATION σ=${SIGMA} warmup=${warmup}  PPL=$PPL  NAN_CHUNKS=$NAN"
}

echo "=== Branch-3 A.2 σ×warmup ABLATION  NODE_IDX=${NODE_IDX}  σ=${SIGMA} ==="

run_cell 200
run_cell 500
run_cell 1000

echo ""
echo "=== NODE ${NODE_IDX} (σ=${SIGMA}) ALL THREE CELLS COMPLETE ==="
