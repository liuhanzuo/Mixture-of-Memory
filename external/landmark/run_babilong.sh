#!/usr/bin/env bash
# Run BABILong eval on the official Landmark ckpt (LLaMA-1-7B tuned).
# Must be run from external/landmark/ OR any dir (uses absolute paths).
# Uses external/landmark_venv (torch2.1.0 + transformers4.28.1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$EXT/.." && pwd)"
LLAMA_DIR="$EXT/landmark-attention/llama"
PY="$EXT/landmark_venv/bin/python"
SCRIPT="$REPO/scripts/run_babilong_landmark.py"

CKPT="${LM_CKPT:-$EXT/landmark_ckpts/landmark_tuned}"
OUTPUT_NAME="${LM_OUTPUT:-landmark_official}"
DEVICE="${LM_DEVICE:-cuda:4}"
TOP_K="${LM_TOPK:-5}"
TASKS="${LM_TASKS:-qa1 qa2 qa5}"
LENGTHS="${LM_LENGTHS:-0k 1k 2k 4k 8k 16k 32k}"
LIMIT="${LM_LIMIT:-100}"

echo "[landmark-babilong] ckpt:    $CKPT"
echo "[landmark-babilong] output:  $OUTPUT_NAME"
echo "[landmark-babilong] device:  $DEVICE"
echo "[landmark-babilong] top_k:   $TOP_K"
echo "[landmark-babilong] tasks:   $TASKS"
echo "[landmark-babilong] lengths: $LENGTHS"
echo "[landmark-babilong] limit:   $LIMIT"

# llama_mem.py uses `from llama_landmark_config import ...` and
# `from ltriton.flash_landmark_attention import ...` — both are in llama/ dir.
# We must run from there so bare `import llama_mem` works.
cd "$LLAMA_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
exec "$PY" "$SCRIPT" \
    --ckpt_path "$CKPT" \
    --output_name "$OUTPUT_NAME" \
    --device "$DEVICE" \
    --top_k "$TOP_K" \
    --tasks $TASKS \
    --lengths $LENGTHS \
    --limit "$LIMIT" \
    --max_new_tokens 20 \
    "$@"
