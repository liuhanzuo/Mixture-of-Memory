#!/bin/bash
# BABILong evaluation for H6 Cross-Attention Memory (dual-gate)
#
# H6 architecture: Llama-3-8B + middle-layer (L16) cross-attention memory
# with LSTM-style dual-gate writeback. 64 slots, read at layers 18,22,26,30.
#
# Usage:
#   ./scripts/run_babilong_h6.sh                    # H6 step 1000
#   ./scripts/run_babilong_h6.sh --variant h6b      # H6b aggressive

set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

export PYTHONPATH="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

PYTHON=/opt/conda/envs/torch-base/bin/python
GPU=5
VARIANT="h6"

# Parse variant argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set checkpoint and output name based on variant
if [ "$VARIANT" = "h6b" ]; then
    CKPT_PATH="outputs/experiment_h6b_dual_gate_aggressive/step_1000.pt"
    OUTPUT_NAME="H6b-step1000"
    echo "============================================"
    echo "BABILong Eval: H6b (aggressive dual-gate)"
    echo "============================================"
else
    CKPT_PATH="outputs/experiment_h6_dual_gate/step_1000.pt"
    OUTPUT_NAME="H6-step1000"
    echo "============================================"
    echo "BABILong Eval: H6 (dual-gate)"
    echo "============================================"
fi

echo "  Checkpoint: $CKPT_PATH"
echo "  Output:     babilong_results/$OUTPUT_NAME"
echo "  GPU:        $GPU"
echo "  Variant:    $VARIANT"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_babilong_h6.py \
    --ckpt_path "$CKPT_PATH" \
    --model_path /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --output_name "$OUTPUT_NAME" \
    --tasks qa1 qa2 qa3 qa4 qa5 \
    --lengths 0k 1k 2k 4k 8k 16k 32k \
    --chunk_size 4096 \
    --num_slots 64 \
    --memory_write_layer 16 \
    --memory_read_layers "18,22,26,30" \
    --memory_init strided \
    --use_dual_gate \
    --forget_bias_init 1.0 \
    --input_bias_init 0.0 \
    --max_new_tokens 20 \
    --device cuda:0 \
    "$@"
