#!/bin/bash
# fix_d_ablation: Test Fix D (gate init 0.5 + entropy aux loss) for routing degeneracy.
#
# Root cause (confirmed by researcher):
#   self.slot_output_gate = nn.Parameter(torch.zeros(())) → tanh(0) = 0
#   → next_hidden = bypass_h + 0 * slot_delta = bypass_h exactly
#   → zero gradient to all selector parameters (Q_sel, slot_keys)
#   → permanent routing degeneracy regardless of fixes A/B/C (upstream of zero multiplier)
#
# Fix D (2026-04-28):
#   D.1: init slot_output_gate = 0.5 → tanh(0.5) ≈ 0.462 → immediate gradient flow
#   D.2: entropy aux loss (weight=0.001) → non-zero gradient at uniform fixed point
#
# This ablation includes ALL prior fixes (A: slot_init_noise, B: learnable slot_keys,
# C: cosine normalization) PLUS Fix D on top.
#
# Same sigma grid as fix_b_ablation: sigma in {0.01, 0.02, 0.05} across 3 nodes.
#
# Usage (launched separately per node):
#     NODE_IDX=0 bash scripts/_run_fix_d_ablation.sh   # b200-2  sigma=0.01
#     NODE_IDX=1 bash scripts/_run_fix_d_ablation.sh   # b200-3  sigma=0.02
#     NODE_IDX=2 bash scripts/_run_fix_d_ablation.sh   # b200-4  sigma=0.05
#
# Run spec: 8xB200, max_chunks=200, max_steps=10000 (enough to check milestones at 100/200/1000/2000).
# Kill criterion: top1_sim < 0.005 at step 200 -> routing still broken.
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

TAG="fix_d_ablation_sigma${SIGMA}_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_d_ablation_sigma${SIGMA}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29550 + NODE_IDX))

# Force HuggingFace to use local files only — bypasses hub validation that
# rejects absolute local paths as invalid repo IDs (transformers 5.5.4 regression).
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_d_ablation  NODE_IDX=${NODE_IDX}  sigma=${SIGMA}  port=${PORT} ==="
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

echo "=== fix_d_ablation NODE_IDX=${NODE_IDX} sigma=${SIGMA} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
