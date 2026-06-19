#!/usr/bin/env bash
# Phase 1 S0: recover Landmark-tuned LLaMA-1-7B from the official weight-diff, then run
# passkey-retrieval eval to reproduce the paper's long-range (32k ~100%) result.
#
# Run from external/landmark/ (this dir). Uses the isolated external/landmark_venv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$(cd "$HERE/.." && pwd)"                       # external/
REPO="$EXT/landmark-attention/llama"                # official repo code (llama_mem, weight_diff)
PY="$EXT/landmark_venv/bin/python"
CKPTS="$EXT/landmark_ckpts"

BASE="${BASE:-$CKPTS/llama1_7b_base}"               # original LLaMA-1-7B (HF format)
WDIFF="${WDIFF:-$CKPTS/wdiff}"                       # epfml/landmark-attention-llama7b-wdiff
TUNED="${TUNED:-$CKPTS/landmark_tuned}"             # output: recovered landmark ckpt

echo "[recover] base=$BASE"
echo "[recover] wdiff=$WDIFF"
echo "[recover] tuned(out)=$TUNED"

cd "$REPO"
# weight_diff.py recover adds the diff back onto the base, runs the naive checksum integrity
# check (default psum 49798.7656 = original LLaMA-1-7B), and saves the recovered ckpt.
"$PY" weight_diff.py recover \
    --path_raw "$BASE" \
    --path_diff "$WDIFF" \
    --path_tuned "$TUNED" \
    --device cpu \
    --test_inference False

echo "[recover] done -> $TUNED"
