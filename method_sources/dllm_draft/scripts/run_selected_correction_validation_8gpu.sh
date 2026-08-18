#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CALIBRATION="$ROOT/outputs/c2_calibration_v0/tasks.jsonl"
SELECTION="$ROOT/ops/artifacts/correction_policy_selection.json"
OUT="$ROOT/outputs/correction_validation_v0"
TASKS="$OUT/tasks.jsonl"
CHECKPOINT="$(cat "$ROOT/ops/artifacts/scaffold_stage1_latest_checkpoint.txt")"

test -s "$CALIBRATION"
test -s "$SELECTION"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

"$ENV_DIR/bin/python" "$ROOT/scripts/build_correction_calibration_set.py" \
  --input "$ROOT/data/scaffold_edu_v0/eval_data.parquet" \
  --output "$TASKS" \
  --manifest "$OUT/tasks.manifest.json" \
  --depth-0-1 24 \
  --depth-2 24 \
  --depth-3-plus 16 \
  --seed 20260725 \
  --exclude-task-file "$CALIBRATION"

IFS=$'\t' read -r POLICY LABEL THRESHOLD < <(
  "$ENV_DIR/bin/python" - "$SELECTION" <<'PY'
import json
import re
import sys

selection = json.load(open(sys.argv[1]))
policy = selection["best_policy"]
label = selection["best_label"]
threshold = ""
if policy != "C0":
    match = re.search(r"_t(\d{3})$", label)
    if match is None:
        raise SystemExit(f"cannot parse threshold from {label}")
    threshold = f"{int(match.group(1)) / 1000:.3f}"
print(f"{policy}\t{label}\t{threshold}")
PY
)
echo "Validating selected correction policy=$POLICY label=$LABEL threshold=$THRESHOLD"

run_arm() {
  local name="$1"
  local policy="$2"
  local threshold="$3"
  local dir="$OUT/$name"
  local correction_args=()
  case "$policy" in
    C0)
      ;;
    C1)
      correction_args=(
        --leaf-remask-fraction 0.05
        --leaf-remask-confidence-threshold "$threshold"
        --leaf-remask-min-age-calls 1
        --max-leaf-remasks 8
        --max-leaf-remasks-per-token 1
      )
      ;;
    C2)
      correction_args=(
        --structural-backtrack-confidence-threshold "$threshold"
        --structural-backtrack-min-age-calls 1
        --max-structural-backtracks 1
        --max-structural-backtracks-per-anchor 1
      )
      ;;
    C3)
      correction_args=(
        --structural-confidence-threshold "$threshold"
        --structural-max-defer-calls 1
      )
      ;;
    *)
      echo "Unsupported correction policy $policy" >&2
      exit 2
      ;;
  esac
  mkdir -p "$dir"
  "$ENV_DIR/bin/torchrun" \
    --standalone \
    --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_scaffold.py" \
    --checkpoint "$CHECKPOINT" \
    --dataset humaneval \
    --data-file "$TASKS" \
    --output-dir "$dir/shards" \
    --max-model-calls 256 \
    --transfer-tokens 1 \
    --seed-function-signature \
    --resume \
    "${correction_args[@]}"
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --expected 64
  "$ENV_DIR/bin/python" "$ROOT/scripts/evaluate_correction_calibration.py" \
    --tasks "$TASKS" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --output "$dir/evaluation.jsonl" \
    --summary "$dir/summary.json"
}

run_arm c0 C0 ""
SELECTED_DIR="$OUT/c0"
if [ "$POLICY" != C0 ]; then
  run_arm selected "$POLICY" "$THRESHOLD"
  SELECTED_DIR="$OUT/selected"
fi

"$ENV_DIR/bin/python" "$ROOT/scripts/compare_correction_validation.py" \
  --baseline-evaluation "$OUT/c0/evaluation.jsonl" \
  --selected-evaluation "$SELECTED_DIR/evaluation.jsonl" \
  --baseline-summary "$OUT/c0/summary.json" \
  --selected-summary "$SELECTED_DIR/summary.json" \
  --policy "$POLICY" \
  --label "$LABEL" \
  --bootstrap-replicates 20000 \
  --seed 20260725 \
  --output "$ROOT/ops/artifacts/correction_validation.json" \
  --markdown-output "$ROOT/ops/artifacts/CORRECTION_VALIDATION.md"

date --iso-8601=seconds \
  >"$ROOT/ops/control/correction_validation.done"
