#!/usr/bin/env bash
# Run the ShortGPT layer-selection (Men et al. 2024) for Paper B on OLMo-2-7B.
# Forward-only Block-Influence pass over a small Dolmino calibration set; picks
# the 16 highest-BI layers to KEEP (drops the 16 lowest-BI). ~2-5 min on 1 GPU
# (or slower on CPU). Writes outputs/shortgpt_layer_selection.json, read by
# scripts/train_olmo2_shortgpt.py.
#
# NEVER calibrates on MMLU / downstream data (uses data/dolmino_now15b.npy).
#
# Usage:
#   bash scripts/_run_shortgpt_select.sh                      # 1 GPU, 128 windows
#   DEVICE=cpu NUM_WINDOWS=64 bash scripts/_run_shortgpt_select.sh   # CPU sanity
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
CALIB_DATA="${CALIB_DATA:-data/dolmino_now15b.npy}"
NUM_WINDOWS="${NUM_WINDOWS:-128}"
KEEP_LAYERS="${KEEP_LAYERS:-16}"
BI_METRIC="${BI_METRIC:-cosine}"
DEVICE="${DEVICE:-auto}"
CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT="${OUTPUT:-outputs/shortgpt_layer_selection.json}"
GPU="${GPU:-0}"

mkdir -p outputs logs
LOG="logs/shortgpt_select.log"

echo "[_run_shortgpt_select] model=$MODEL_PATH calib=$CALIB_DATA windows=$NUM_WINDOWS keep=$KEEP_LAYERS metric=$BI_METRIC device=$DEVICE -> $OUTPUT"

CVD=""
[ "$DEVICE" != "cpu" ] && CVD="CUDA_VISIBLE_DEVICES=$GPU"

env $CVD "$PYTHON_BIN" scripts/shortgpt_select_layers.py \
  --model_path "$MODEL_PATH" \
  --calib_data "$CALIB_DATA" \
  --num_calib_windows "$NUM_WINDOWS" \
  --calib_seq_len "$CALIB_SEQ_LEN" \
  --keep_layers "$KEEP_LAYERS" \
  --bi_metric "$BI_METRIC" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output "$OUTPUT" 2>&1 | tee "$LOG"

echo "[_run_shortgpt_select] done; selection in $OUTPUT (log $LOG)"
