#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
MODEL_DIR="${DREAM_CODER_MODEL_DIR:-$ROOT/models/Dream-Coder-v0-Instruct-7B}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$ENV_DIR/bin/python" "$ROOT/scripts/smoke_dream_coder.py" \
  --model-path "$MODEL_DIR" \
  --output "$ROOT/ops/artifacts/dream_coder_smoke.json" \
  --steps "${DREAM_SMOKE_STEPS:-32}" \
  --max-new-tokens "${DREAM_SMOKE_MAX_NEW_TOKENS:-32}"

date --iso-8601=seconds >"$ROOT/ops/control/dream_coder_smoke.done"

