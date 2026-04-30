#!/bin/bash
# fix_g_ablation: Test Fix G (SKRL — Pairwise Slot-Key Repulsion Loss) for K_sel routing degeneracy.
#
# Root cause (confirmed 2026-04-29, researcher report 20260429_fix_g_root_cause.md):
#   Fixes A–F all failed because their gradient signals depend on slot *content* being diverse,
#   but slot content stays near-uniform until routing becomes non-uniform — a chicken-and-egg
#   deadlock. Specifically:
#   - Fix F centered STE: gradient magnitude O(sigma) ≈ O(0.02) = 100× too small at init.
#   - Load-balance / entropy losses: zero or near-zero gradient w.r.t. slot_keys directly.
#   All 6 prior fixes left top1_sim ≈ 1/512 = 0.00195 (symmetric fixed point) indefinitely.
#
# Fix G (2026-04-29): Pairwise slot-key repulsion loss (SKRL).
#   Acts directly on slot_keys geometry, independent of slot content.
#   loss = mean cosine similarity of randomly sampled pairs of F.normalize(slot_keys).
#   At symmetric fixed point: gradient is O(1) — pushes every key away from its neighbours.
#   Does NOT require slot content to be diverse first.
#
#   Implementation:
#     selector.slot_key_diversity_loss(num_pairs=512) — in TopKSelector
#     config.skrl_weight — weight on the SKRL term in the aux loss
#     layer.py: aux["skrl"] = skrl_loss * cfg.skrl_weight  (in return_aux_losses block)
#     _collect_aux_loss: sums "skrl" key alongside "load_balance" and "entropy"
#     [SKRL_DIAG]: logs mean_pairwise_cos every 200 fwd steps
#
# Includes ALL prior fixes (A through F):
# A: slot_init_noise, B: learnable slot_keys, C: cosine normalization,
# D.1: gate init=0.5, D.2: entropy_aux_weight, E: full-scale M_sel_hidden projection,
# F: centered STE gradient multiplier. G: SKRL (this fix).
#
# Ablation: sigma fixed at 0.02 (Fix F's longest-running sigma), sweep skrl_weight.
# NODE_IDX 0/1/2 → skrl_weight 0.001 / 0.01 / 0.1
#
# Usage (launched separately per node):
#     NODE_IDX=0 bash scripts/_run_fix_g_ablation.sh   # b200-2  skrl_weight=0.001
#     NODE_IDX=1 bash scripts/_run_fix_g_ablation.sh   # b200-3  skrl_weight=0.01
#     NODE_IDX=2 bash scripts/_run_fix_g_ablation.sh   # b200-4  skrl_weight=0.1
#
# Run spec: 8×B200, max_chunks=200, max_steps=10000
# Kill criterion: top1_sim < 0.005 at step 200 → routing still broken.
# Success criterion: top1_sim > 0.05 at step 1000 AND niah_acc > 0.05 at step 2000.
# SKRL diagnostic: [SKRL_DIAG fwd=200] mean_pairwise_cos should drop below -0.001 by fwd=400.

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

NODE_IDX="${NODE_IDX:-0}"

# sigma fixed at 0.02 (best performer in Fix F ablation); sweep skrl_weight
SIGMA=0.02
case "$NODE_IDX" in
  0) SKRL_WEIGHT=0.001 ;;
  1) SKRL_WEIGHT=0.01  ;;
  2) SKRL_WEIGHT=0.1   ;;
  *) echo "ERROR: NODE_IDX must be 0,1,2 (got $NODE_IDX)"; exit 2 ;;
esac

MODEL="${MODEL:-/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TAG="fix_g_ablation_skrl${SKRL_WEIGHT}_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_g_ablation_skrl${SKRL_WEIGHT}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29580 + NODE_IDX))

# Force HuggingFace to use local files only — bypasses hub validation that
# rejects absolute local paths as invalid repo IDs (transformers 5.5.4 regression).
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_g_ablation  NODE_IDX=${NODE_IDX}  sigma=${SIGMA}  skrl_weight=${SKRL_WEIGHT}  port=${PORT} ==="
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
    --skrl_weight "${SKRL_WEIGHT}" \
    --max_steps 10000 \
    --lr 3e-4 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --slot_init random \
    --slot_init_noise "${SIGMA}" \
    --niah_mix_fraction 0.10 \
    --niah_max_N 16 \
    --swa_window 512 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "=== fix_g_ablation NODE_IDX=${NODE_IDX} skrl_weight=${SKRL_WEIGHT} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
