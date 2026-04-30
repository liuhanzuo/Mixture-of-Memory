#!/bin/bash
# fix_i_ablation: Test Fix I (hidden_to_slot actually included in optimizer) for K_sel routing degeneracy.
#
# Root cause chain (confirmed 2026-04-29):
#   Fixes A–H all failed to break routing degeneracy floor (top1_sim ≈ 1/512 = 0.00195).
#   Fix H diagnosis: GATE_GRAD_DIAG confirmed hidden_to_slot.weight.grad_norm=None even when
#   --unfreeze_hidden_to_slot was passed. Root cause: _mem_space_params() never included
#   hidden_to_slot parameters regardless of the flag — making --unfreeze_hidden_to_slot a no-op.
#
# Fix I (2026-04-29): Conditionally include hidden_to_slot in _mem_space_params()
#   Modified _mem_space_params() in scripts/train_mem_space_pg19.py:
#     if not getattr(wrapper.config, 'hidden_to_slot_frozen', True):
#         for p in wrapper.hidden_to_slot.parameters():
#             params.append(p)
#   This makes the write projection actually trainable when --unfreeze_hidden_to_slot is passed.
#   Key check: GATE_GRAD_DIAG should show hidden_to_slot.weight.grad_norm != None at step ≤ 20.
#
# Includes ALL prior fixes (A through H):
# A: slot_init_noise, B: learnable slot_keys, C: cosine normalization,
# D.1: gate init=0.5, D.2: entropy_aux_weight, E: full-scale M_sel_hidden projection,
# F: centered STE gradient multiplier, G: SKRL disabled,
# H: Differentiable soft routing proxy + slot norm clipping.
# I: hidden_to_slot actually included in optimizer (this fix).
#
# Single configuration — 3 nodes running same config for reliability.
# NODE_IDX 0/1/2 → b200-2/b200-3/b200-4
#
# Usage (launched separately per node):
#     NODE_IDX=0 bash scripts/_run_fix_i_ablation.sh   # b200-2
#     NODE_IDX=1 bash scripts/_run_fix_i_ablation.sh   # b200-3
#     NODE_IDX=2 bash scripts/_run_fix_i_ablation.sh   # b200-4
#
# Run spec: 8×B200, max_chunks=200, max_steps=10000
# Key diagnostic: [GATE_GRAD_DIAG] hidden_to_slot.weight.grad_norm — must be non-None at step ≤ 20.
# Key diagnostic: [QUERY_DIAG] top1_sim_mean — should rise above 0.005 within 500 steps.
# Kill criterion: hidden_to_slot.weight.grad_norm=None at any GATE_GRAD_DIAG → immediate kill (Fix I not working).
# Kill criterion: top1_sim < 0.005 at step 500.
# Success criterion: top1_sim > 0.05 at step 1000 (unblocks req_20260427_102400_scale_up_N1024).

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

TAG="fix_i_ablation_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_i_ablation_node${NODE_IDX}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29580 + NODE_IDX))

# Force HuggingFace to use local files only
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_i_ablation  NODE_IDX=${NODE_IDX}  port=${PORT} ==="
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
    --slot_init_noise 0.02 \
    --niah_mix_fraction 0.10 \
    --niah_max_N 16 \
    --swa_window 512 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "=== fix_i_ablation NODE_IDX=${NODE_IDX} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
