#!/bin/bash
# fix_j_ablation: Test Fix I + Fix J-A together.
#
# Fix J-A (2026-04-29): Remove slots.detach() from soft-proxy einsum in layer.py:499
#   slots.detach() blocked gradient flow through hidden_to_slot even when Fix I added it
#   to the optimizer. With both fixes, hidden_to_slot.weight.grad_norm should be non-None.
#
# Ablation sweep: 3 nodes with different slot_init_noise sigma values
#   NODE_IDX 0 → b200-2: sigma=0.01
#   NODE_IDX 1 → b200-3: sigma=0.02
#   NODE_IDX 2 → b200-4: sigma=0.05
#
# Success criterion: hidden_to_slot.weight.grad_norm != None at n_done=5
# Also track: trainable_with_grad=224/224 (was 128/224 in fix_i)
# Also track: top1_sim_mean rising above 0.005 within 500 steps
#
# Usage:
#     NODE_IDX=0 bash scripts/_run_fix_j_ablation.sh   # b200-2
#     NODE_IDX=1 bash scripts/_run_fix_j_ablation.sh   # b200-3
#     NODE_IDX=2 bash scripts/_run_fix_j_ablation.sh   # b200-4

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

NODE_IDX="${NODE_IDX:-0}"

MODEL="${MODEL:-/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TAG="fix_j_ablation_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_j_ablation_node${NODE_IDX}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29590 + NODE_IDX))

# Select sigma by node
if   [ "$NODE_IDX" = "0" ]; then SIGMA=0.01
elif [ "$NODE_IDX" = "1" ]; then SIGMA=0.02
else                                  SIGMA=0.05
fi

# Force HuggingFace to use local files only
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_j_ablation  NODE_IDX=${NODE_IDX}  sigma=${SIGMA}  port=${PORT} ==="
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT_DIR"

torchrun --nproc_per_node=8 --master_port=$PORT \
    "$PROJECT_DIR/scripts/train_mem_space_pg19.py" \
    --model "$MODEL" \
    --data "$DATA" \
    --max_chunks 200 \
    --skip_chunks 200 \
    --seq_len 4096 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --writeback_warmup_steps 500 \
    --load_balance_weight 0.01 \
    --entropy_aux_weight 0.001 \
    --skrl_weight 0.0 \
    --max_steps 10000 \
    --lr 3e-4 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --slot_init random \
    --slot_init_noise ${SIGMA} \
    --niah_mix_fraction 0.10 \
    --niah_max_N 16 \
    --swa_window 512 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "=== fix_j_ablation NODE_IDX=${NODE_IDX} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
