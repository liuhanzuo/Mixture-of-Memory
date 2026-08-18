#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
VANILLA_OUT="${SEMANTIC_VANILLA_OUT:-$ROOT/outputs/semantic_scaffold_vanilla_humaneval}"
SCAFFOLD_OUT="${SEMANTIC_SCAFFOLD_OUT:-$ROOT/outputs/semantic_scaffold_medium_humaneval}"
OUT="${SEMANTIC_GATE_OUT:-$ROOT/outputs/semantic_scaffold_gate}"
SUCCESS="${SEMANTIC_GATE_SUCCESS:-$ROOT/ops/control/semantic_scaffold_gate.done}"

test -s "$VANILLA_OUT/eval_results.json"
test -s "$SCAFFOLD_OUT/eval_results.json"
mkdir -p "$OUT"

"$ENV_DIR/bin/python" "$ROOT/scripts/evaluate_semantic_scaffold_gate.py" \
  --vanilla "$VANILLA_OUT" \
  --scaffold "$SCAFFOLD_OUT" \
  --output "$OUT/report.json"

date --iso-8601=seconds >"$SUCCESS"
