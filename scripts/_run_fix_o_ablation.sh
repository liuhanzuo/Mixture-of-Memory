#!/bin/bash
# fix_o_ablation: Fix routing collapse caused by temperature=10.0 amplifying LM gradient 100× vs SKRL.
#
# Root cause of fix_n_ablation routing collapse:
#   top1_sim = 1/N = 0.002 floor for all 820–1070 steps across all nodes.
#   mean_pairwise_cos POSITIVE throughout — slot keys clustering, not diverging.
#   Root cause: temperature = 10.0 hardcoded in selector.py:152.
#     LM gradient ∝ T → at T=10, LM:SKRL ratio = 100:1.
#     SKRL repels keys but LM clusters them 100× harder → bounded oscillation.
#     SKRL IS working (b200-2 cos briefly -0.003) but insufficient vs T=10.
#
# Fix O changes vs fix_n_ablation:
#   - temperature 10.0 → 1.0 (selector_temperature CLI arg, Fix O 2026-04-29)
#   - LM:SKRL ratio drops from 100:1 → 10:1 at skrl_weight=0.10
#   - Ablation: b200-2 adds entropy_aux_weight=0.001 to isolate its effect
#
# Full fix stack (inherited):
#   Fix I:   hidden_to_slot in _mem_space_params() optimizer group
#   Fix J-A: remove slots.detach() from soft-proxy einsum (layer.py:499)
#   Fix K:   strided_token slot init + _detach_banks carry-over
#   Fix L-1: adaptive M_sel_hidden norm clip (input side)
#   Fix L-2: per-param grad clip 0.1 for slot_to_hidden/hidden_to_slot
#   Fix L-3: WRITEBACK_DIAG interval 200→50
#   Fix M-1: slot_delta norm clip to bypass_h scale (output side)
#   Fix N:   SKRL re-enabled (variable weight), load_balance 10× lower
#   Fix O:   selector_temperature 10.0 → 1.0 (LM:SKRL ratio 100:1 → 10:1)  ← NEW
#
# Ablation sweep: T=1.0 fixed, varied SKRL weights and entropy
#   NODE_IDX 0 → b200-2: skrl=0.10, entropy=0.001  (T=1.0, isolate entropy effect)
#   NODE_IDX 1 → b200-3: skrl=0.05, entropy=0.0    (T=1.0, moderate SKRL no entropy)
#   NODE_IDX 2 → b200-4: skrl=0.10, entropy=0.0    (T=1.0, strongest SKRL no entropy)
#
# Success criterion (same as fix_n):
#   step 200: mean_pairwise_cos < -0.002  (slot keys spreading apart)
#   step 300: top1_sim_mean > 0.003       (above 1/N floor)
#   step 500: top1_sim_mean > 0.005
#   step 1000: top1_sim_mean > 0.05      → unblocks req_20260427_102400_scale_up_N1024
#
# Usage:
#     NODE_IDX=0 bash scripts/_run_fix_o_ablation.sh   # b200-2
#     NODE_IDX=1 bash scripts/_run_fix_o_ablation.sh   # b200-3
#     NODE_IDX=2 bash scripts/_run_fix_o_ablation.sh   # b200-4

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

TAG="fix_o_ablation_node${NODE_IDX}"
OUTPUT_DIR="$PROJECT_DIR/outputs/fix_o_ablation_node${NODE_IDX}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${TAG}_${TIMESTAMP}.log"

PORT=$((29800 + NODE_IDX))

# Fixed sigma for all nodes
SIGMA=0.01

# Fix O: temperature 1.0 for all nodes (was hardcoded 10.0 in selector.py)
TEMPERATURE=1.0

# Select SKRL weight and entropy by node
if   [ "$NODE_IDX" = "0" ]; then SKRL_W=0.10; ENTROPY_W=0.001
elif [ "$NODE_IDX" = "1" ]; then SKRL_W=0.05; ENTROPY_W=0.0
else                              SKRL_W=0.10; ENTROPY_W=0.0
fi

# Force HuggingFace to use local files only
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== fix_o_ablation  NODE_IDX=${NODE_IDX}  temperature=${TEMPERATURE}  sigma=${SIGMA}  skrl_weight=${SKRL_W}  entropy=${ENTROPY_W}  port=${PORT} ==="
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
    --load_balance_weight 0.001 \
    --entropy_aux_weight ${ENTROPY_W} \
    --skrl_weight ${SKRL_W} \
    --selector_temperature ${TEMPERATURE} \
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

echo "=== fix_o_ablation NODE_IDX=${NODE_IDX} DONE ==="
cat "$OUTPUT_DIR/eval_results.json" 2>/dev/null || echo "(no eval_results.json)"
