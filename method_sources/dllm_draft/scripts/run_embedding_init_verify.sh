#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
MODEL_DIR="${DREAM_CODER_MODEL_DIR:-$ROOT/models/Dream-Coder-v0-Instruct-7B}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT"

"$ENV_DIR/bin/python" "$ROOT/scripts/verify_embedding_initialization.py" \
  --model-path "$MODEL_DIR" \
  --output "$ROOT/ops/artifacts/embedding_initialization.json"

date --iso-8601=seconds >"$ROOT/ops/control/embedding_initialization.done"

