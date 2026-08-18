#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
if [ -n "${SCAFFOLD_DECODE_CHECKPOINT:-}" ]; then
  CHECKPOINT="$SCAFFOLD_DECODE_CHECKPOINT"
elif [ -f "$ROOT/ops/artifacts/scaffold_stage1_latest_checkpoint.txt" ]; then
  CHECKPOINT="$(cat "$ROOT/ops/artifacts/scaffold_stage1_latest_checkpoint.txt")"
else
  CHECKPOINT="$ROOT/outputs/scaffold_sft_smoke/global_step_3"
fi

test -d "$CHECKPOINT"

export PYTHONPATH="$ROOT:$ROOT/vendor/DreamOn:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$ENV_DIR/bin/python" "$ROOT/scripts/run_scaffold_decode_smoke.py" \
  --checkpoint "$CHECKPOINT" \
  --output "$ROOT/ops/artifacts/scaffold_decode_smoke.json"

date --iso-8601=seconds >"$ROOT/ops/control/scaffold_decode_smoke.done"
