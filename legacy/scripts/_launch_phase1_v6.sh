#!/bin/bash
# Phase 1: Better Gate Initialization Launch Script for Node3
# Target: Fix 4000x gradient dilution + gate saturation issue
# Node: pighzliu@28.89.19.134 (8×B200, ~180GB each)

set -e

# ===================================================================
# Configuration
# ===================================================================

NODE="pighzliu@28.89.19.134"
BASE_MODEL="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/models/Llama--Llama2-7b"
DATA_PATH="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/pg19_train.jsonl"
OUTPUT_DIR="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/phase1_gate_init_v6"

# ✅ Phase 1: Proper gate initialization
GATE_BIAS_INIT=0.0  # sigmoid(0) = 0.5, max gradient flow

# Sparse memory config
NUM_SLOTS=128
WINDOW_SIZE=256
TOP_K=8
EMA_ALPHA=0.1

# Training config
BATCH_SIZE=1
GRAD_ACCUM=8          # Effective batch size = 1 × 8 = 8
LR=5e-5
MAX_STEPS=1000
SAVE_STEPS=100        # Checkpoints at 100, 200, 300, ..., 1000
LOGGING_STEPS=10
WARMUP_STEPS=50

# ===================================================================
# Print Configuration
# ===================================================================

echo "=============================================="
echo "  Phase 1: Better Gate Initialization"
echo "=============================================="
echo ""
echo "📦 Model:"
echo "  Base model: $BASE_MODEL"
echo "  Output dir: $OUTPUT_DIR"
echo ""
echo "🧠 Sparse Memory:"
echo "  Num slots: $NUM_SLOTS"
echo "  Window size: $WINDOW_SIZE"
echo "  Top-k: $TOP_K"
echo "  EMA alpha: $EMA_ALPHA"
echo ""
echo "🚪 Gate Initialization (Phase 1 Fix):"
echo "  gate_bias_init: $GATE_BIAS_INIT"
echo "  Expected sigmoid: $(python3 -c "import math; print(f'{math.exp(0)/(math.exp(0)+1):.4f}')")"
echo "  Expected sigmoid gradient: $(python3 -c "import math; s=math.exp(0)/(math.exp(0)+1); print(f'{s*(1-s):.4f}')")"
echo ""
echo "📊 Training:"
echo "  Node: $NODE (8×B200)"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM grad accum = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "  Learning rate: $LR"
echo "  Max steps: $MAX_STEPS"
echo "  Save steps: $SAVE_STEPS"
echo "  Logging steps: $LOGGING_STEPS"
echo ""
echo "⏱️  Estimated Duration:"
echo "  To step 100: ~20-30 minutes"
echo "  To step 300: ~1-1.5 hours (comparable to v5 baseline)"
echo "  To step 1000: ~4-6 hours"
echo ""
echo "=============================================="
echo ""

# ===================================================================
# Check Node Availability
# ===================================================================

echo "Checking node availability..."
if ssh $NODE "echo 'Node is reachable'" 2>/dev/null; then
    echo "✅ Node is reachable"
else
    echo "❌ ERROR: Cannot reach node $NODE"
    exit 1
fi

# ===================================================================
# Create Output Directory
# ===================================================================

echo ""
echo "Creating output directory on remote node..."
ssh $NODE "mkdir -p $OUTPUT_DIR" || {
    echo "❌ ERROR: Failed to create output directory"
    exit 1
}
echo "✅ Output directory ready"

# ===================================================================
# Launch Training
# ===================================================================

echo ""
echo "=============================================="
echo "  🚀 Launching Training on Node3"
echo "=============================================="
echo ""

ssh $NODE "cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory && \
torchrun --nproc_per_node=8 scripts/train_gated_sparse_memory.py \
  --model_name_or_path '$BASE_MODEL' \
  --data_path '$DATA_PATH' \
  --output_dir '$OUTPUT_DIR' \
  --max_seq_length 2048 \
  --num_slots $NUM_SLOTS \
  --window_size $WINDOW_SIZE \
  --top_k $TOP_K \
  --ema_alpha $EMA_ALPHA \
  --gate_bias_init $GATE_BIAS_INIT \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --learning_rate $LR \
  --max_steps $MAX_STEPS \
  --warmup_steps $WARMUP_STEPS \
  --save_steps $SAVE_STEPS \
  --logging_steps $LOGGING_STEPS \
  --save_total_limit 3 \
  --bf16 \
  --gradient_checkpointing \
  --ddp_find_unused_parameters"

echo ""
echo "=============================================="
echo "  ✅ Training Launched Successfully"
echo "=============================================="
echo ""
echo "📝 Monitoring logs:"
echo "  ssh $NODE 'tail -f $OUTPUT_DIR/trainer_log.jsonl'"
echo ""
echo "📊 Checkpoint schedule:"
for step in $(seq 100 $SAVE_STEPS $MAX_STEPS); do
    echo "  Step $step: $OUTPUT_DIR/checkpoint-$step"
done
echo ""
