#!/usr/bin/env bash
# Phase 1 S0: run passkey-retrieval eval on the recovered Landmark ckpt (+ base for contrast).
# Run from external/landmark/. Uses isolated external/landmark_venv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$(cd "$HERE/.." && pwd)"
REPO="$EXT/landmark-attention/llama"
PY="$EXT/landmark_venv/bin/python"
CKPTS="$EXT/landmark_ckpts"

export LM_BASE="${LM_BASE:-$CKPTS/llama1_7b_base}"
export LM_TUNED="${LM_TUNED:-$CKPTS/landmark_tuned}"
export LM_CACHE="${LM_CACHE:-$HERE/hf-cache}"
export LM_MODELS="${LM_MODELS:-base,mem}"
export LM_TOPK="${LM_TOPK:-5}"
export LM_NTESTS="${LM_NTESTS:-50}"
export LM_OUT="${LM_OUT:-$HERE/passkey_results.csv}"
export LM_BASE_DEVICE="${LM_BASE_DEVICE:-cuda:0}"
export LM_MEM_DEVICE="${LM_MEM_DEVICE:-cuda:0}"
# LM_NVALUES / LM_SEED can be overridden via env.

# run_passkey.py imports llama_mem from the official repo dir -> run from there.
cd "$REPO"
exec "$PY" "$HERE/run_passkey.py"
