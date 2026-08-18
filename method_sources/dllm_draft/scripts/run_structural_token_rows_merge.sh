#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
BASE="$(cat "$ROOT/ops/artifacts/semantic_scaffold_selected_checkpoint.txt")"
ADAPTER="${TOKEN_ROW_ADAPTER:-$ROOT/outputs/structural_token_rows_native_smoke/global_step_4}"
OUT="${TOKEN_ROW_MERGED_OUT:-$ROOT/outputs/structural_token_rows_native_merged}"
POINTER="${TOKEN_ROW_POINTER:-$ROOT/ops/artifacts/structural_token_rows_native_merged_checkpoint.txt}"
SUCCESS="${TOKEN_ROW_MERGE_SUCCESS:-$ROOT/ops/control/structural_token_rows_native_merge.done}"

test -s "$ADAPTER/adapter_model.safetensors"
"$ROOT/scripts/setup_peft019_overlay.sh"
export PYTHONPATH="$ROOT/.peft019:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
"$ENV_DIR/bin/python" "$ROOT/scripts/merge_trainable_token_rows.py" \
  --base "$BASE" \
  --adapter "$ADAPTER" \
  --output "$OUT"
test -s "$OUT/model.safetensors.index.json"
test -s "$OUT/token_row_merge_manifest.json"

printf '%s\n' "$OUT" \
  >"$POINTER"
date --iso-8601=seconds >"$SUCCESS"
