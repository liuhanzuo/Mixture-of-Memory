#!/bin/bash
# fix_h_ablation: Test Fix H (Differentiable Soft Routing Proxy) for K_sel routing degeneracy.
#
# Root cause (confirmed 2026-04-29, researcher report 20260429_fix_h_proposal.md):
#   Fixes A–G all failed to break the routing degeneracy deadlock:
#   - Fix G (SKRL): Slot keys were ALREADY orthogonal at init (pairwise cos≈0 on S^127).
#     SKRL solved a non-problem.
#   - True root cause: Fix F STE has M_sel_centered≈0 because all 512 slots initialized from
#     the same hidden_pool_mean. Centering subtracts the dominant common component, leaving
#     near-zero gradient to Q_sel. No gradient → no routing improvement → slots never diverge.
#
# Fix H (2026-04-29): Differentiable Soft Routing Proxy
#   Hard forward: use actual selected slot content (correct behavior, unchanged).
#   Backward: route gradient through soft weighted sum over ALL slots.
#     M_sel_hidden_soft = slot_to_hidden(einsum("bn,bnd->bd", scores, slots.detach()))
#     STE recombination: M_sel_hidden = M_sel_hard.detach() + (M_sel_soft - M_sel_soft.detach())
#   Gradient: d(loss)/d(scores[i]) = d(loss)/d(M_sel_soft) · slot_to_hidden(slots[i])  — O(1), non-zero.
#   Secondary: slot norm clipping in memory_bank.write() prevents 32-layer EMA compounding NaN.
#   SKRL disabled (config.skrl_weight=0.0, confirmed ineffective).
#
# Includes ALL prior fixes (A through F):
# A: slot_init_noise, B: learnable slot_keys, C: cosine normalization,
# D.1: gate init=0.5, D.2: entropy_aux_weight, E: full-scale M_sel_hidden projection,
# F: centered STE gradient multiplier (superseded by Fix H's soft proxy).
# G: SKRL disabled (skrl_weight=0.0 default).
# H: Differentiable soft routing proxy + slot norm clipping (this fix).
#
# Single configuration — no ablation sweep needed.
# NODE_IDX 0/1/2 → b200-2/b200-3/b200-4 (same ports as fix_g)
#
# Usage (launched separately per node):
#     NODE_IDX=0 bash scripts/_run_fix_h_ablation.sh   # b200-2
#     NODE_IDX=1 bash scripts/_run_fix_h_ablation.sh   # b200-3
#     NODE_IDX=2 bash scripts/_run_fix_h_ablation.sh   # b200-4
#
# Run spec: 8×B200, max_chunks=200, max_steps=10000
# Key diagnostic: [QUERY_DIAG] top1_sim_mean — should rise above 0.005 within 500 steps.
# Kill criterion: top1_sim < 0.005 at step 500 (more generous than Fix G since soft proxy needs
#   more steps to warm up, but should show clear upward trend by step 200).
# Success criterion: top1_sim > 0.01 at step 1000.

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

TAG="fix_h_ablation_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_h_ablation_node${NODE_IDX}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29580 + NODE_IDX))

# Force HuggingFace to use local files only
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_h_ablation  NODE_IDX=${NODE_IDX}  port=${PORT} ==="
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

echo "=== fix_h_ablation NODE_IDX=${NODE_IDX} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
