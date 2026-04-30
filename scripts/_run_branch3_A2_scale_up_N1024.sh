#!/bin/bash
# Run 3/3: Branch-3 A.2 N=1024 / k=128 scale-up on local 8×H20.
#
# Purpose: test whether the winner σ=0.02 + warmup=500 + shared_bank config
# keeps working when we double the slot budget.  Stage-2b previously failed
# at N=1024 k=128 (PPL=426) under the OLD σ=1.0 + warmup=0 regime — now that
# H1 is cured, does doubling capacity actually help PPL below 1.9051?
#
# Target: local node, 8× H20 (97.8 GiB each), ≈8 min wall at 200 steps
# (memory cost: 2× slots, unchanged seq_len=4096; slot attention O(Nk) ≈ 2×
# baseline, well within H20 envelope).
# Expected outcome: PPL ≤ 1.9 (below current winner) if scale helps; PPL ≈
# 1.9 would mean we've saturated retrieval capacity at N=512.

set -e
set -u

# Local canonical tree (zwfy6 is the edit copy; canonical shared mount for
# training scripts is wzc1, but the local H20 node also keeps its own copy).
PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

MODEL="${MODEL:-$PROJECT_DIR/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/branch3_A2_scale_up_N1024_k128}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

MAX_CHUNKS=200
SEQ_LEN=4096
NUM_SLOTS=1024
TOP_K=128
TRAIN_STEPS=200
NUM_GPUS=8

echo "=== Branch-3 A.2 SCALE-UP N=${NUM_SLOTS} k=${TOP_K} (winner σ=0.02 config) ==="
echo "Target: PPL ≤ 1.9 (beat Exp-A v2 @ 1.9051)"
echo ""

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$NUM_GPUS --master_port=29540 \
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
    --writeback_warmup_steps 500 \
    --load_balance_weight 0.01 \
    --max_train_steps $TRAIN_STEPS \
    --lr 1e-3 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --slot_init random \
    --slot_init_noise 0.02 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== FINAL eval_results.json ==="
cat "$OUTPUT_DIR/eval_results.json"
PPL=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['ppl'])")
NAN=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/eval_results.json'))['nan_chunks'])")
echo "SCALE-UP N=${NUM_SLOTS} k=${TOP_K}  PPL=$PPL  NAN_CHUNKS=$NAN"
