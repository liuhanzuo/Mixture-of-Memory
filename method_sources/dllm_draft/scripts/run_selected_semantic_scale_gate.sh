#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CALIBRATION="$ROOT/outputs/semantic_lora_scale_calibration"
POINTER="$ROOT/ops/artifacts/semantic_scaffold_selected_checkpoint.txt"
OUT="$ROOT/outputs/semantic_scaffold_selected_scale"
SUCCESS="$ROOT/ops/control/semantic_scaffold_selected_scale.done"

test -s "$CALIBRATION/report.json"
"$ENV_DIR/bin/python" "$ROOT/scripts/select_semantic_lora_scale.py" \
  --calibration-root "$CALIBRATION" \
  --pointer "$POINTER" \
  --output "$OUT/selection.json" \
  --minimum-screen-plus 0.50

date --iso-8601=seconds >"$SUCCESS"
