#!/bin/bash
# Run 1/3: Ship-config SCHEDULE-MATCH extended (Branch-3 A.2 winner @ 1000 steps)
#
# Purpose: validate the 1.9051 PPL from A_v2 is not a 200-step early-advantage
# artifact. Extend training to 1000 steps with the winner config.
#
# Target: b200-1 (28.89.17.143), 8×B200, ≈30 min wall.
# Expected outcome: PPL stays ≤ 2.70, ideally ≤ 2.0.

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

MODEL="${MODEL:-$PROJECT_DIR/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/branch3_A2_schedule_match_1000}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

MAX_CHUNKS=1000
SEQ_LEN=4096
NUM_SLOTS=512
TOP_K=64
TRAIN_STEPS=1000
NUM_GPUS=8

echo "=== Branch-3 A.2 SCHEDULE-MATCH (1000 steps, winner config) ==="
echo "Target: beat PPL=1.9051 and verify advantage persists at longer schedule"
echo ""

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$NUM_GPUS --master_port=29521 \
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
echo "SCHEDULE-MATCH PPL=$PPL  NAN_CHUNKS=$NAN"
