#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
MODEL_DIR="${DREAM_CODER_MODEL_DIR:-$ROOT/models/Dream-Coder-v0-Instruct-7B}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$ENV_DIR/bin/python" "$ROOT/scripts/smoke_dream_coder_cpu_load.py" \
  --model-path "$MODEL_DIR" \
  --output "$ROOT/ops/artifacts/dream_coder_cpu_load.json"

date --iso-8601=seconds >"$ROOT/ops/control/dream_coder_cpu_load.done"

