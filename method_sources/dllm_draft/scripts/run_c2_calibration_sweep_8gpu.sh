#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
OUT="$ROOT/outputs/c2_calibration_v0"
TASKS="$OUT/tasks.jsonl"
MANIFEST="$OUT/tasks.manifest.json"
CHECKPOINT="$(cat "$ROOT/ops/artifacts/scaffold_stage1_latest_checkpoint.txt")"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "$OUT"
"$ENV_DIR/bin/python" "$ROOT/scripts/build_correction_calibration_set.py" \
  --input "$ROOT/data/scaffold_edu_v0/eval_data.parquet" \
  --output "$TASKS" \
  --manifest "$MANIFEST"

run_arm() {
  local label="$1"
  local threshold="$2"
  local dir="$OUT/$label"
  local c2_args=()
  if [ -n "$threshold" ]; then
    c2_args=(
      --structural-backtrack-confidence-threshold "$threshold"
      --structural-backtrack-min-age-calls 1
      --max-structural-backtracks 1
      --max-structural-backtracks-per-anchor 1
    )
  fi
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
    "${c2_args[@]}"
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --expected 32
  "$ENV_DIR/bin/python" "$ROOT/scripts/evaluate_correction_calibration.py" \
    --tasks "$TASKS" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --output "$dir/evaluation.jsonl" \
    --summary "$dir/summary.json"
}

run_arm c0 ""
run_arm c2_t005 0.05
run_arm c2_t010 0.10
run_arm c2_t020 0.20
run_arm c2_t030 0.30

"$ENV_DIR/bin/python" "$ROOT/scripts/select_correction_calibration.py" \
  --policy C2 \
  --run c0="$OUT/c0/summary.json" \
  --run c2_t005="$OUT/c2_t005/summary.json" \
  --run c2_t010="$OUT/c2_t010/summary.json" \
  --run c2_t020="$OUT/c2_t020/summary.json" \
  --run c2_t030="$OUT/c2_t030/summary.json" \
  --max-nfe-ratio 1.25 \
  --max-token-ratio 1.35 \
  --output "$ROOT/ops/artifacts/c2_calibration_selection.json" \
  --markdown-output "$ROOT/ops/artifacts/C2_CALIBRATION_RESULTS.md"

date --iso-8601=seconds \
  >"$ROOT/ops/control/c2_calibration_sweep.done"
