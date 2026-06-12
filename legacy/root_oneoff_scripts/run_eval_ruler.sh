#!/bin/bash
# RULER CWE baseline evaluation for Qwen3-8B at 64K context (1 GPU)
#
# Usage:
#   bash run_eval_ruler.sh                   # Full CWE eval at 64K
#   bash run_eval_ruler.sh --limit 5         # Quick sanity check (5 examples)
#   TASK=ruler bash run_eval_ruler.sh        # All 13 RULER tasks
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  GPU to use (default: 0)
#   MODEL_PATH            Model path (default: models/Qwen--Qwen3-8b)
#   OUTPUT_DIR            Results dir (default: eval_results/ruler_cwe_qwen3-8b_64k)
#   TASK                  RULER task (default: ruler_cwe)
#   CONTEXT_LEN           Context length (default: 65536)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-models/Qwen--Qwen3-8b}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/ruler_cwe_qwen3-8b_64k}"
TASK="${TASK:-ruler_cwe}"
CONTEXT_LEN="${CONTEXT_LEN:-65536}"

echo "=== RULER Evaluation Launcher ==="
echo "  GPU:    ${CUDA_VISIBLE_DEVICES}"
echo "  Model:  ${MODEL_PATH}"
echo "  Task:   ${TASK}"
echo "  Ctx:    ${CONTEXT_LEN}"
echo ""

python3 eval_ruler_baseline.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --task "${TASK}" \
    --context_length "${CONTEXT_LEN}" \
    "$@"
