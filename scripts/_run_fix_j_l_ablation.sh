#!/bin/bash
# fix_j_l_ablation: Test Fix I + Fix J-A + Fix K + Fix L together.
#
# Extends fix_j_ablation by:
#   Fix K (2026-04-29): strided_token slot init + _detach_banks carry-over (in train script)
#   Fix L-1 (2026-04-29): Adaptive M_sel_hidden norm clip in layer.py (prevents slot_to_hidden
#     weight growth from generating memory tokens 20-44x above hidden_states scale)
#   Fix L-2 (2026-04-29): Per-param grad clip 0.1 for slot_to_hidden/hidden_to_slot in train script
#   Fix L-3 (2026-04-29): WRITEBACK_DIAG logging 200→50 steps for earlier explosion detection
#
# Root cause of fix_j_ablation NaN: slot_to_hidden weight growth (lr=1e-3) amplifies slot norms
# (max 128 via Fix H) to M_sel_hidden norms 1368-2055 at step 1033-1125 (expected ~32).
# Fix L-1 clips M_sel_hidden to hidden_states norm scale — prevents positive-feedback spiral.
#
# Ablation sweep: same 3 sigma values as fix_j_ablation
#   NODE_IDX 0 → b200-2: sigma=0.01
#   NODE_IDX 1 → b200-3: sigma=0.02
#   NODE_IDX 2 → b200-4: sigma=0.05
#
# Success criterion:
#   step 100:  WRITEBACK_DIAG M_sel_hidden_norm_mean < 50 (Fix L-1 active)
#   step 500:  top1_sim_mean > 0.005
#   step 1000: top1_sim_mean > 0.05 → unblocks req_20260427_102400_scale_up_N1024
#
# Usage:
#     NODE_IDX=0 bash scripts/_run_fix_j_l_ablation.sh   # b200-2
#     NODE_IDX=1 bash scripts/_run_fix_j_l_ablation.sh   # b200-3
#     NODE_IDX=2 bash scripts/_run_fix_j_l_ablation.sh   # b200-4

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

TAG="fix_j_l_ablation_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_j_l_ablation_node${NODE_IDX}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29600 + NODE_IDX))

# Select sigma by node (same as fix_j_ablation)
if   [ "$NODE_IDX" = "0" ]; then SIGMA=0.01
elif [ "$NODE_IDX" = "1" ]; then SIGMA=0.02
else                                  SIGMA=0.05
fi

# Force HuggingFace to use local files only
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_j_l_ablation  NODE_IDX=${NODE_IDX}  sigma=${SIGMA}  port=${PORT} ==="
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
    --slot_init strided_token \
    --slot_init_noise ${SIGMA} \
    --niah_mix_fraction 0.10 \
    --niah_max_N 16 \
    --swa_window 512 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "=== fix_j_l_ablation NODE_IDX=${NODE_IDX} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
