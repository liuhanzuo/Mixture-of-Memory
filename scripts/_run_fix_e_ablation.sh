#!/bin/bash
# fix_e_ablation: Test Fix E (remove w_gathered attenuation from M_sel_hidden projection)
#                 for routing degeneracy.
#
# Root cause (confirmed 2026-04-28, main agent independent analysis after researcher stall):
#   Original code: M_sel_hidden = slot_to_hidden(M_sel_slot * w_gathered)
#   At uniform routing init: w_gathered ≈ 1/512 = 0.00195 (STE weight for each of k slots)
#   slot_to_hidden has std=0.02 init → ||M_sel_hidden|| ≈ 0.0016 vs ||H|| ≈ 64 → ratio=2.5e-5
#   Slot tokens are ~40,000× smaller than text tokens in cross-attention.
#   → slot_delta = ext_h[:,k_slots:,:] - bypass_h ≈ 0
#   → d(loss)/d(slot_output_gate) ∝ slot_delta ≈ 0 → gate stays frozen at 0.462891
#   → d(loss)/d(Q_sel) ≈ 0, d(loss)/d(slot_keys) ≈ 0 → permanent routing degeneracy
#
# Evidence (WRITEBACK_DIAG across fix_d_ablation, 430 steps):
#   alpha(tanh_output_gate) = 0.462891 CONSTANT at steps 97/204/298/427
#   → slot_output_gate parameter received effectively zero gradient from Adam lr=3e-4
#
# Fix E (2026-04-28):
#   Remove w_gathered from M_sel_hidden projection forward path.
#   Preserve STE gradient via additive zero-valued correction:
#     M_sel_hidden = slot_to_hidden(M_sel_slot)                         # full scale
#     M_sel_hidden += M_sel_hidden.detach() * (w_gathered - w_gathered.detach())
#   Forward: correction term = 0 (no numerical change to transformer input)
#   Backward: d/d(w_gathered) = M_sel_hidden.detach() ≠ 0
#             → gradient flows Q_sel/slot_keys through ste_weights → scores → logits
#
# Sanity check (numerical):
#   OLD: ||M_sel_hidden|| = 0.001602, ratio to H = 0.000025 (40,000× attenuation)
#   NEW: ||M_sel_hidden|| = 0.820047, ratio to H = 0.012813 (512× improvement, exactly 1/N removed)
#
# This ablation includes ALL prior fixes (A: slot_init_noise, B: learnable slot_keys,
# C: cosine normalization, D.1: gate init=0.5, D.2: entropy_aux_weight) PLUS Fix E on top.
#
# Same sigma grid as fix_d_ablation: sigma in {0.01, 0.02, 0.05} across 3 nodes.
#
# Usage (launched separately per node):
#     NODE_IDX=0 bash scripts/_run_fix_e_ablation.sh   # b200-2  sigma=0.01
#     NODE_IDX=1 bash scripts/_run_fix_e_ablation.sh   # b200-3  sigma=0.02
#     NODE_IDX=2 bash scripts/_run_fix_e_ablation.sh   # b200-4  sigma=0.05
#
# Run spec: 8xB200, max_chunks=200, max_steps=10000 (enough to check milestones at 100/200/1000/2000).
# Kill criterion: top1_sim < 0.005 at step 200 → routing still broken.
# Success criterion: top1_sim > 0.05 at step 1000 AND niah_acc > 0.05 at step 2000.

set -e
set -u

PROJECT_DIR="${PROJECT_DIR:-/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base

NODE_IDX="${NODE_IDX:-0}"

case "$NODE_IDX" in
  0) SIGMA=0.01 ;;
  1) SIGMA=0.02 ;;
  2) SIGMA=0.05 ;;
  *) echo "ERROR: NODE_IDX must be 0,1,2 (got $NODE_IDX)"; exit 2 ;;
esac

MODEL="${MODEL:-/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b}"
DATA="${DATA:-$PROJECT_DIR/data/pg19_chunks_llama3.npy}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TAG="fix_e_ablation_sigma${SIGMA}_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_e_ablation_sigma${SIGMA}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29560 + NODE_IDX))

# Force HuggingFace to use local files only — bypasses hub validation that
# rejects absolute local paths as invalid repo IDs (transformers 5.5.4 regression).
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_e_ablation  NODE_IDX=${NODE_IDX}  sigma=${SIGMA}  port=${PORT} ==="
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

echo "=== fix_e_ablation NODE_IDX=${NODE_IDX} sigma=${SIGMA} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
