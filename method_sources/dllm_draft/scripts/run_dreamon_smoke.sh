#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
MODEL_DIR="${DREAMON_MODEL_DIR:-$ROOT/models/DreamOn-v0-7B}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$ENV_DIR/bin/python" "$ROOT/scripts/smoke_dreamon.py" \
  --model-path "$MODEL_DIR" \
  --output "$ROOT/ops/artifacts/dreamon_smoke.json" \
  --initial-masks 4 \
  --max-new-tokens 32

date --iso-8601=seconds >"$ROOT/ops/control/dreamon_smoke.done"

