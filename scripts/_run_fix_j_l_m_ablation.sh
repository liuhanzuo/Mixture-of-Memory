#!/bin/bash
# fix_j_l_m_ablation: Test Fix I + Fix J-A + Fix K + Fix L + Fix M-1 together.
#
# Extends fix_j_l_ablation by:
#   Fix M-1 (2026-04-29): slot_delta output-side per-token norm clip in layer.py
#     Root cause of fix_j_l_ablation chronic PPL spikes: slot_delta (output injection) was
#     unclipped. slot_delta_max=7.97 × alpha=0.462 × 32 layers = 117 effective residual shift.
#     Fix L-1 guards INPUT side (M_sel_hidden). Fix M-1 guards OUTPUT side (slot_delta).
#     One-directional clip: slot_delta norm capped to bypass_h norm scale (clamp max=1.0).
#
# Full fix stack:
#   Fix I:   hidden_to_slot in _mem_space_params() optimizer group
#   Fix J-A: remove slots.detach() from soft-proxy einsum (layer.py:499)
#   Fix K:   strided_token slot init + _detach_banks carry-over
#   Fix L-1: adaptive M_sel_hidden norm clip (input side)
#   Fix L-2: per-param grad clip 0.1 for slot_to_hidden/hidden_to_slot
#   Fix L-3: WRITEBACK_DIAG interval 200→50
#   Fix M-1: slot_delta norm clip to bypass_h scale (output side) ← NEW
#
# Ablation sweep: same 3 sigma values
#   NODE_IDX 0 → b200-2: sigma=0.01
#   NODE_IDX 1 → b200-3: sigma=0.02
#   NODE_IDX 2 → b200-4: sigma=0.05
#
# Success criterion:
#   step 100:  WRITEBACK_DIAG M_sel_hidden_norm_mean < 50 + slot_delta_max < 1.0
#   step 500:  top1_sim_mean > 0.005
#   step 1000: top1_sim_mean > 0.05 → unblocks req_20260427_102400_scale_up_N1024
#
# Usage:
#     NODE_IDX=0 bash scripts/_run_fix_j_l_m_ablation.sh   # b200-2
#     NODE_IDX=1 bash scripts/_run_fix_j_l_m_ablation.sh   # b200-3
#     NODE_IDX=2 bash scripts/_run_fix_j_l_m_ablation.sh   # b200-4

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

TAG="fix_j_l_m_ablation_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_j_l_m_ablation_node${NODE_IDX}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29700 + NODE_IDX))

# Select sigma by node (same as fix_j_l_ablation)
if   [ "$NODE_IDX" = "0" ]; then SIGMA=0.01
elif [ "$NODE_IDX" = "1" ]; then SIGMA=0.02
else                                  SIGMA=0.05
fi

# Force HuggingFace to use local files only
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_j_l_m_ablation  NODE_IDX=${NODE_IDX}  sigma=${SIGMA}  port=${PORT} ==="
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

echo "=== fix_j_l_m_ablation NODE_IDX=${NODE_IDX} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
