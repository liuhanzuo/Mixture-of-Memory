#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CHECKPOINT="${CAPACITY_CHECKPOINT:-$ROOT/outputs/scaffold_sft_stage1/global_step_4465}"
OUT="${CAPACITY_OUTPUT_ROOT:-$ROOT/outputs/runtime_capacity_screen}"
TASKS="$OUT/tasks.jsonl"
EXPECTED="${CAPACITY_SCREEN_SIZE:-64}"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"

mkdir -p "$OUT"
"$ENV_DIR/bin/python" "$ROOT/scripts/build_runtime_capacity_screen.py" \
  --input "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  --output "$TASKS" \
  --manifest "$OUT/tasks.manifest.json" \
  --size "$EXPECTED" \
  --seed 20260805

run_config() {
  local label="$1"
  shift
  local dir="$OUT/$label"
  if [[ "${CAPACITY_RERUN:-0}" == "1" ]]; then
    rm -rf "$dir"
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
    --max-model-calls 512 \
    --transfer-tokens 1 \
    --runtime-config-label "$label" \
    --resume \
    "$@"
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --expected "$EXPECTED"
  "$ENV_DIR/bin/python" - "$dir/metrics.jsonl" "$dir/summary.json" <<'PY'
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text().splitlines()
    if line.strip()
]
successful = [row["process"] for row in rows if row.get("process")]
failed = [row.get("failure_process") or {} for row in rows if row.get("error")]
nfes = [
    float((row.get("process") or row.get("failure_process"))["nfe"])
    for row in rows
]
cumulative = [
    float(
        (row.get("process") or row.get("failure_process"))[
            "cumulative_model_tokens"
        ]
    )
    for row in rows
    if (
        row.get("process") or row.get("failure_process")
    ).get("cumulative_model_tokens") is not None
]
reasons = Counter(
    (row.get("failure_process") or {}).get(
        "termination_reason", "unknown"
    )
    for row in rows
    if row.get("error")
)
summary = {
    "rows": len(rows),
    "successes": len(successful),
    "failures": len(failed),
    "failure_rate": len(failed) / len(rows),
    "termination_reasons": dict(reasons),
    "all_task_mean_nfe": statistics.mean(nfes),
    "all_task_median_nfe": statistics.median(nfes),
    "successful_mean_nfe": (
        statistics.mean(p["nfe"] for p in successful)
        if successful else None
    ),
    "successful_mean_cumulative_tokens": (
        statistics.mean(
            float(row["process"]["cumulative_model_tokens"])
            for row in rows
            if row.get("process")
        )
        if successful else None
    ),
    "all_task_mean_cumulative_tokens": (
        statistics.mean(cumulative) if cumulative else None
    ),
    "mean_capacity_metrics": {
        key: statistics.mean(
            float((row.get("process") or row.get("failure_process") or {}).get(key, 0))
            for row in rows
        )
        for key in (
            "line_capacity_hits",
            "token_capacity_hits",
            "depth_capacity_hits",
            "total_line_capacity_hits",
            "module_expand_suppressed",
            "expand_budget_hits",
            "maximum_tree_depth",
            "maximum_total_lines",
            "maximum_body_lines",
            "maximum_tokens_per_hole",
        )
    },
}
Path(sys.argv[2]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

should_run() {
  local label="$1"
  local requested=",${CAPACITY_CONFIGS:-tiny,small,medium,large},"
  [[ "$requested" == *",$label,"* ]]
}

if should_run tiny; then
run_config tiny \
  --initial-root-slots 2 \
  --initial-body-slots 2 \
  --initial-statement-masks 1 \
  --initial-function-header-masks 1 \
  --initial-loop-header-masks 1 \
  --initial-condition-masks 1 \
  --max-tree-depth 1 \
  --max-lines-per-body 2 \
  --max-total-lines 16 \
  --max-tokens-per-hole 2 \
  --max-expansions 32 \
  --no-module-expand
fi

if should_run small; then
run_config small \
  --initial-root-slots 1 \
  --initial-body-slots 2 \
  --initial-statement-masks 2 \
  --initial-function-header-masks 2 \
  --initial-loop-header-masks 2 \
  --initial-condition-masks 2 \
  --max-tree-depth 2 \
  --max-lines-per-body 4 \
  --max-total-lines 32 \
  --max-tokens-per-hole 8 \
  --max-expansions 128
fi

if should_run medium; then
run_config medium \
  --initial-root-slots 1 \
  --initial-body-slots 2 \
  --initial-statement-masks 4 \
  --initial-function-header-masks 4 \
  --initial-loop-header-masks 4 \
  --initial-condition-masks 3 \
  --max-tree-depth 4 \
  --max-lines-per-body 16 \
  --max-total-lines 64 \
  --max-tokens-per-hole 32 \
  --max-expansions 256
fi

if should_run large; then
run_config large \
  --initial-root-slots 1 \
  --initial-body-slots 2 \
  --initial-statement-masks 4 \
  --initial-function-header-masks 4 \
  --initial-loop-header-masks 4 \
  --initial-condition-masks 3 \
  --max-tree-depth 16 \
  --max-lines-per-body 128 \
  --max-total-lines 1024 \
  --max-tokens-per-hole 512 \
  --max-expansions 512
fi

"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = {
    label: json.load(open(root / label / "summary.json"))
    for label in ("tiny", "small", "medium", "large")
    if (root / label / "summary.json").exists()
}
(root / "comparison.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

date --iso-8601=seconds \
  >"$ROOT/ops/control/runtime_capacity_screen.done"
