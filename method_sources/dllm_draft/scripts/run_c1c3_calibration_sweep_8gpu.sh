#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
C2_OUT="$ROOT/outputs/c2_calibration_v0"
OUT="$ROOT/outputs/c1c3_calibration_v0"
TASKS="$C2_OUT/tasks.jsonl"
CHECKPOINT="$(cat "$ROOT/ops/artifacts/scaffold_stage1_latest_checkpoint.txt")"

test -s "$TASKS"
test -s "$C2_OUT/c0/summary.json"
export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

run_arm() {
  local label="$1"
  local policy="$2"
  local threshold="$3"
  local dir="$OUT/$label"
  local correction_args=()
  case "$policy" in
    C1)
      correction_args=(
        --leaf-remask-fraction 0.05
        --leaf-remask-confidence-threshold "$threshold"
        --leaf-remask-min-age-calls 1
        --max-leaf-remasks 8
        --max-leaf-remasks-per-token 1
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
    --expected 32
  "$ENV_DIR/bin/python" "$ROOT/scripts/evaluate_correction_calibration.py" \
    --tasks "$TASKS" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --output "$dir/evaluation.jsonl" \
    --summary "$dir/summary.json"
}

run_arm c1_t005 C1 0.05
run_arm c1_t010 C1 0.10
run_arm c1_t020 C1 0.20
run_arm c3_t005 C3 0.05
run_arm c3_t010 C3 0.10
run_arm c3_t020 C3 0.20

"$ENV_DIR/bin/python" "$ROOT/scripts/select_correction_calibration.py" \
  --policy C1 \
  --run c0="$C2_OUT/c0/summary.json" \
  --run c1_t005="$OUT/c1_t005/summary.json" \
  --run c1_t010="$OUT/c1_t010/summary.json" \
  --run c1_t020="$OUT/c1_t020/summary.json" \
  --max-nfe-ratio 1.25 \
  --max-token-ratio 1.35 \
  --output "$ROOT/ops/artifacts/c1_calibration_selection.json" \
  --markdown-output "$ROOT/ops/artifacts/C1_CALIBRATION_RESULTS.md"

"$ENV_DIR/bin/python" "$ROOT/scripts/select_correction_calibration.py" \
  --policy C3 \
  --run c0="$C2_OUT/c0/summary.json" \
  --run c3_t005="$OUT/c3_t005/summary.json" \
  --run c3_t010="$OUT/c3_t010/summary.json" \
  --run c3_t020="$OUT/c3_t020/summary.json" \
  --max-nfe-ratio 1.25 \
  --max-token-ratio 1.35 \
  --output "$ROOT/ops/artifacts/c3_calibration_selection.json" \
  --markdown-output "$ROOT/ops/artifacts/C3_CALIBRATION_RESULTS.md"

"$ENV_DIR/bin/python" "$ROOT/scripts/summarize_correction_policies.py" \
  --report "$ROOT/ops/artifacts/c1_calibration_selection.json" \
  --report "$ROOT/ops/artifacts/c2_calibration_selection.json" \
  --report "$ROOT/ops/artifacts/c3_calibration_selection.json" \
  --output "$ROOT/ops/artifacts/correction_policy_selection.json" \
  --markdown-output "$ROOT/ops/artifacts/CORRECTION_POLICY_SELECTION.md"

date --iso-8601=seconds \
  >"$ROOT/ops/control/c1c3_calibration_sweep.done"
