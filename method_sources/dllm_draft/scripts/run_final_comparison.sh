#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DREAM_ENV_DIR:-$ROOT/.venv_dream}/bin/python"

test -f "$ROOT/outputs/schedule_only_evalplus_full/humaneval/eval_results.json"
test -f "$ROOT/outputs/schedule_only_evalplus_full/mbpp/eval_results.json"
test -f "$ROOT/outputs/schedule_only_humaneval_nfe64/eval_results.json"
test -f "$ROOT/outputs/schedule_only_humaneval_nfe128/eval_results.json"
test -f "$ROOT/outputs/dream_coder_humaneval_nfe128/eval_results.json"
test -f "$ROOT/outputs/plain_evalplus_full/humaneval/eval_results.json"
test -f "$ROOT/outputs/plain_evalplus_full/mbpp/eval_results.json"
test -f "$ROOT/outputs/plain_humaneval_nfe64/eval_results.json"
test -f "$ROOT/outputs/plain_humaneval_nfe128/eval_results.json"

write_manifest() {
  local name="$1" mode="$2" checkpoint_file="$3" launcher="$4" metrics="$5"
  local run_id="$6" run_log="$7"
  shift 7
  local provenance_args=()
  local path
  for path in "$@"; do
    provenance_args+=(--provenance-file "$path")
  done
  "$PYTHON" "$ROOT/scripts/write_experiment_manifest.py" \
    --name "$name" \
    --mode "$mode" \
    --checkpoint "$(cat "$checkpoint_file")" \
    --launcher "$launcher" \
    --run-id "$run_id" \
    --run-log "$run_log" \
    --training-metrics "$metrics" \
    --train-data "$ROOT/data/scaffold_edu_v0/train_data.parquet" \
    --eval-data "$ROOT/data/scaffold_edu_v0/eval_data.parquet" \
    "${provenance_args[@]}" \
    --output "$ROOT/ops/artifacts/manifests/$name.json"
}

write_manifest \
  scaffold_stage1 hierarchical \
  "$ROOT/ops/artifacts/scaffold_stage1_latest_checkpoint.txt" \
  "$ROOT/scripts/run_scaffold_sft_stage1_8gpu.sh" \
  "$ROOT/outputs/scaffold_sft_stage1/training_metrics.jsonl" \
  SCAFFOLD-SFT-STAGE1-8GPU-001 \
  "$ROOT/ops/logs/SCAFFOLD-SFT-STAGE1-8GPU-001.log" \
  "$ROOT/ops/artifacts/scaffold_throughput_sweep.json"
write_manifest \
  schedule_only_stage1 schedule_only \
  "$ROOT/ops/artifacts/schedule_only_stage1_latest_checkpoint.txt" \
  "$ROOT/scripts/run_schedule_only_sft_stage1_8gpu.sh" \
  "$ROOT/outputs/schedule_only_sft_stage1/training_metrics.jsonl" \
  SCHEDULE-ONLY-SFT-STAGE1-001 \
  "$ROOT/ops/logs/SCHEDULE-ONLY-SFT-STAGE1-001.log"
write_manifest \
  plain_stage1 plain \
  "$ROOT/ops/artifacts/plain_stage1_latest_checkpoint.txt" \
  "$ROOT/scripts/run_plain_sft_stage1_8gpu.sh" \
  "$ROOT/outputs/plain_sft_stage1/training_metrics.jsonl" \
  PLAIN-SFT-STAGE1-001 \
  "$ROOT/ops/logs/PLAIN-SFT-STAGE1-001.log" \
  "$ROOT/ops/artifacts/plain_bucketed_throughput_sweep.json" \
  "$ROOT/ops/artifacts/plain_bucketed_micro_batch.txt"

"$PYTHON" "$ROOT/scripts/compare_evalplus_runs.py" \
  --run scaffold="$ROOT/outputs/evalplus_full/humaneval" \
  --run dream512="$ROOT/outputs/dream_coder_evalplus_full/humaneval" \
  --run dream64="$ROOT/outputs/dream_coder_humaneval_nfe64" \
  --run dream128="$ROOT/outputs/dream_coder_humaneval_nfe128" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/humaneval" \
  --run schedule64="$ROOT/outputs/schedule_only_humaneval_nfe64" \
  --run schedule128="$ROOT/outputs/schedule_only_humaneval_nfe128" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/humaneval" \
  --run plain64="$ROOT/outputs/plain_humaneval_nfe64" \
  --run plain128="$ROOT/outputs/plain_humaneval_nfe128" \
  --output "$ROOT/ops/artifacts/final_humaneval_comparison.json"

"$PYTHON" "$ROOT/scripts/compare_evalplus_runs.py" \
  --run scaffold="$ROOT/outputs/evalplus_full/mbpp" \
  --run dream512="$ROOT/outputs/dream_coder_evalplus_full/mbpp" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/mbpp" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/mbpp" \
  --output "$ROOT/ops/artifacts/final_mbpp_comparison.json"

"$PYTHON" "$ROOT/scripts/analyze_eval_by_depth.py" \
  --dataset-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  --run scaffold="$ROOT/outputs/evalplus_full/humaneval/eval_results.json" \
  --run dream512="$ROOT/outputs/dream_coder_evalplus_full/humaneval/eval_results.json" \
  --run dream64="$ROOT/outputs/dream_coder_humaneval_nfe64/eval_results.json" \
  --run dream128="$ROOT/outputs/dream_coder_humaneval_nfe128/eval_results.json" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/humaneval/eval_results.json" \
  --run schedule64="$ROOT/outputs/schedule_only_humaneval_nfe64/eval_results.json" \
  --run schedule128="$ROOT/outputs/schedule_only_humaneval_nfe128/eval_results.json" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/humaneval/eval_results.json" \
  --run plain64="$ROOT/outputs/plain_humaneval_nfe64/eval_results.json" \
  --run plain128="$ROOT/outputs/plain_humaneval_nfe128/eval_results.json" \
  --output "$ROOT/ops/artifacts/final_humaneval_by_depth.json"

"$PYTHON" "$ROOT/scripts/analyze_eval_by_depth.py" \
  --dataset-file "$ROOT/data/evalplus/MbppPlus-v0.2.0.jsonl" \
  --run scaffold="$ROOT/outputs/evalplus_full/mbpp/eval_results.json" \
  --run dream512="$ROOT/outputs/dream_coder_evalplus_full/mbpp/eval_results.json" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/mbpp/eval_results.json" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/mbpp/eval_results.json" \
  --output "$ROOT/ops/artifacts/final_mbpp_by_depth.json"

"$PYTHON" "$ROOT/scripts/summarize_training_metrics.py" \
  --run scaffold="$ROOT/outputs/scaffold_sft_stage1/training_metrics.jsonl" \
  --run schedule="$ROOT/outputs/schedule_only_sft_stage1/training_metrics.jsonl" \
  --run plain="$ROOT/outputs/plain_sft_stage1/training_metrics.jsonl" \
  --warmup-records 2 \
  --output "$ROOT/ops/artifacts/final_training_efficiency.json"

"$PYTHON" "$ROOT/scripts/summarize_run_history.py" \
  --history "$ROOT/ops/history.tsv" \
  --run scaffold=SCAFFOLD-SFT-STAGE1-8GPU-001 \
  --run schedule=SCHEDULE-ONLY-SFT-STAGE1-001 \
  --run plain=PLAIN-SFT-STAGE1-001 \
  --gpu-count 8 \
  --output "$ROOT/ops/artifacts/final_training_run_costs.json"

"$PYTHON" "$ROOT/scripts/analyze_training_loss.py" \
  --run scaffold="$ROOT/ops/logs/SCAFFOLD-SFT-STAGE1-8GPU-001.log" \
  --run schedule="$ROOT/ops/logs/SCHEDULE-ONLY-SFT-STAGE1-001.log" \
  --run plain="$ROOT/ops/logs/PLAIN-SFT-STAGE1-001.log" \
  --window-size 100 \
  --output "$ROOT/ops/artifacts/final_training_loss.json"

"$PYTHON" "$ROOT/scripts/analyze_failure_taxonomy.py" \
  --run scaffold="$ROOT/outputs/evalplus_full/humaneval" \
  --run dream512="$ROOT/outputs/dream_coder_evalplus_full/humaneval" \
  --run dream64="$ROOT/outputs/dream_coder_humaneval_nfe64" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/humaneval" \
  --run schedule64="$ROOT/outputs/schedule_only_humaneval_nfe64" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/humaneval" \
  --run plain64="$ROOT/outputs/plain_humaneval_nfe64" \
  --output "$ROOT/ops/artifacts/final_humaneval_failure_taxonomy.json"

"$PYTHON" "$ROOT/scripts/analyze_failure_taxonomy.py" \
  --run scaffold="$ROOT/outputs/evalplus_full/mbpp" \
  --run dream512="$ROOT/outputs/dream_coder_evalplus_full/mbpp" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/mbpp" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/mbpp" \
  --output "$ROOT/ops/artifacts/final_mbpp_failure_taxonomy.json"

"$PYTHON" "$ROOT/scripts/analyze_paired_eval.py" \
  --metric plus \
  --run scaffold="$ROOT/outputs/evalplus_full/humaneval/eval_results.json" \
  --run dream64="$ROOT/outputs/dream_coder_humaneval_nfe64/eval_results.json" \
  --run dream128="$ROOT/outputs/dream_coder_humaneval_nfe128/eval_results.json" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/humaneval/eval_results.json" \
  --run schedule64="$ROOT/outputs/schedule_only_humaneval_nfe64/eval_results.json" \
  --run schedule128="$ROOT/outputs/schedule_only_humaneval_nfe128/eval_results.json" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/humaneval/eval_results.json" \
  --run plain64="$ROOT/outputs/plain_humaneval_nfe64/eval_results.json" \
  --run plain128="$ROOT/outputs/plain_humaneval_nfe128/eval_results.json" \
  --pair schedule512,plain512 \
  --pair schedule128,plain128 \
  --pair schedule64,plain64 \
  --pair schedule128,dream128 \
  --pair scaffold,schedule64 \
  --pair scaffold,dream64 \
  --bootstrap-replicates 20000 \
  --seed 20260724 \
  --output "$ROOT/ops/artifacts/final_humaneval_paired.json"

"$PYTHON" "$ROOT/scripts/analyze_paired_eval.py" \
  --metric plus \
  --run scaffold="$ROOT/outputs/evalplus_full/mbpp/eval_results.json" \
  --run dream512="$ROOT/outputs/dream_coder_evalplus_full/mbpp/eval_results.json" \
  --run schedule512="$ROOT/outputs/schedule_only_evalplus_full/mbpp/eval_results.json" \
  --run plain512="$ROOT/outputs/plain_evalplus_full/mbpp/eval_results.json" \
  --pair schedule512,plain512 \
  --pair scaffold,schedule512 \
  --pair schedule512,dream512 \
  --bootstrap-replicates 20000 \
  --seed 20260724 \
  --output "$ROOT/ops/artifacts/final_mbpp_paired.json"

"$PYTHON" "$ROOT/scripts/render_final_report.py" \
  --humaneval-comparison \
  "$ROOT/ops/artifacts/final_humaneval_comparison.json" \
  --mbpp-comparison \
  "$ROOT/ops/artifacts/final_mbpp_comparison.json" \
  --humaneval-depth \
  "$ROOT/ops/artifacts/final_humaneval_by_depth.json" \
  --mbpp-depth \
  "$ROOT/ops/artifacts/final_mbpp_by_depth.json" \
  --training-efficiency \
  "$ROOT/ops/artifacts/final_training_efficiency.json" \
  --training-run-costs \
  "$ROOT/ops/artifacts/final_training_run_costs.json" \
  --training-loss \
  "$ROOT/ops/artifacts/final_training_loss.json" \
  --humaneval-failures \
  "$ROOT/ops/artifacts/final_humaneval_failure_taxonomy.json" \
  --mbpp-failures \
  "$ROOT/ops/artifacts/final_mbpp_failure_taxonomy.json" \
  --humaneval-paired \
  "$ROOT/ops/artifacts/final_humaneval_paired.json" \
  --mbpp-paired \
  "$ROOT/ops/artifacts/final_mbpp_paired.json" \
  --output "$ROOT/ops/artifacts/FINAL_COMPARISON.md" \
  --decision-output "$ROOT/ops/artifacts/g1_pivot_decision.json"

date --iso-8601=seconds \
  >"$ROOT/ops/control/final_comparison.done"
