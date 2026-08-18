#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DREAM_ENV_DIR:-$ROOT/.venv_dream}/bin/python"
OUT="$ROOT/ops/artifacts"

"$PYTHON" "$ROOT/scripts/compare_evalplus_runs.py" \
  --run base_raw="$ROOT/outputs/recovery_base_raw_humaneval" \
  --run base_plain1="$ROOT/outputs/recovery_base_plain_1ep_humaneval" \
  --run base_plain5="$ROOT/outputs/plain_evalplus_full/humaneval" \
  --run instruct_raw="$ROOT/outputs/dream_coder_evalplus_full/humaneval" \
  --run instruct_plain1="$ROOT/outputs/recovery_instruct_plain_1ep_humaneval" \
  --run instruct_highnoise1="$ROOT/outputs/recovery_instruct_highnoise_1ep_humaneval" \
  --output "$OUT/recovery_humaneval_comparison.json"

"$PYTHON" "$ROOT/scripts/analyze_paired_eval.py" \
  --metric plus \
  --run base_raw="$ROOT/outputs/recovery_base_raw_humaneval/eval_results.json" \
  --run base_plain1="$ROOT/outputs/recovery_base_plain_1ep_humaneval/eval_results.json" \
  --run base_plain5="$ROOT/outputs/plain_evalplus_full/humaneval/eval_results.json" \
  --run instruct_raw="$ROOT/outputs/dream_coder_evalplus_full/humaneval/eval_results.json" \
  --run instruct_plain1="$ROOT/outputs/recovery_instruct_plain_1ep_humaneval/eval_results.json" \
  --run instruct_highnoise1="$ROOT/outputs/recovery_instruct_highnoise_1ep_humaneval/eval_results.json" \
  --pair base_plain1,base_raw \
  --pair base_plain5,base_plain1 \
  --pair instruct_plain1,instruct_raw \
  --pair instruct_highnoise1,instruct_plain1 \
  --bootstrap-replicates 20000 \
  --seed 20260727 \
  --output "$OUT/recovery_humaneval_paired.json"

"$PYTHON" "$ROOT/scripts/render_recovery_report.py" \
  --comparison "$OUT/recovery_humaneval_comparison.json" \
  --paired "$OUT/recovery_humaneval_paired.json" \
  --output "$OUT/recovery_diagnostic.json" \
  --markdown-output "$OUT/RECOVERY_DIAGNOSTIC.md"

date --iso-8601=seconds >"$ROOT/ops/control/recovery_comparison.done"
