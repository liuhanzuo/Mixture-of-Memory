#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
RUN="$ROOT/outputs/semantic_scaffold_selected_vanilla_humaneval"
REPORT="$ROOT/outputs/semantic_scaffold_selected_scale/vanilla_gate.json"
SUCCESS="$ROOT/ops/control/semantic_scaffold_selected_vanilla_gate.done"

test -s "$RUN/eval_results.json"
"$ENV_DIR/bin/python" "$ROOT/scripts/check_semantic_vanilla_gate.py" \
  --run "$RUN" \
  --output "$REPORT" \
  --minimum-plus 0.45
date --iso-8601=seconds >"$SUCCESS"
