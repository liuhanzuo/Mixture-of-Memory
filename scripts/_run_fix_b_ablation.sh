#!/bin/bash
# fix_b_ablation: Test Fix B (learnable slot_keys nn.Parameter) for K_sel routing degeneracy.
#
# Fix history:
#   Fix A (2026-04-28): slot_init_noise=1.0 default in config.py
#   Fix C (2026-04-28): cosine normalisation (F.normalize q/k, temperature=10.0) in selector.py
#   Fix B (2026-04-28): standalone slot_keys = nn.Parameter(torch.randn(N, S)*0.1) in selector.py,
#                        replacing K_sel(slots). K_sel frozen for checkpoint compat.
#
# key_fix_ablation (Fix A+C only) failed: top1_sim plateau 0.002060-0.002136 for 4400 steps.
# This run adds Fix B on top of A+C.
#
# Same σ grid as key_fix_ablation: σ ∈ {0.01, 0.02, 0.05} across 3 nodes.
#
# Usage (launched separately per node):
#     NODE_IDX=0 bash scripts/_run_fix_b_ablation.sh   # b200-2  σ=0.01
#     NODE_IDX=1 bash scripts/_run_fix_b_ablation.sh   # b200-3  σ=0.02
#     NODE_IDX=2 bash scripts/_run_fix_b_ablation.sh   # b200-4  σ=0.05
#
# Run spec: 8×B200, max_chunks=200, max_steps=10000 (enough to check milestones at 100/200/1000/2000).
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

TAG="fix_b_ablation_sigma${SIGMA}_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_b_ablation_sigma${SIGMA}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29540 + NODE_IDX))

# Force HuggingFace to use local files only — bypasses hub validation that
# rejects absolute local paths as invalid repo IDs (transformers 5.5.4 regression).
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_b_ablation  NODE_IDX=${NODE_IDX}  σ=${SIGMA}  port=${PORT} ==="
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

echo "=== fix_b_ablation NODE_IDX=${NODE_IDX} σ=${SIGMA} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
