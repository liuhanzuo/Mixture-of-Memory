#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CHECKPOINT="$(cat "$ROOT/ops/artifacts/semantic_scaffold_selected_checkpoint.txt")"
TASKS="$ROOT/outputs/runtime_capacity_screen/tasks.jsonl"
OUT="$ROOT/outputs/semantic_scaffold_seeded_length_calibration"
EXPECTED=64

test -s "$CHECKPOINT/model.safetensors.index.json"
test -s "$TASKS"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HUMANEVAL_OVERRIDE_PATH="$TASKS"

run_length() {
  local label="$1" masks="$2"
  local dir="$OUT/$label"
  "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_scaffold.py" \
    --checkpoint "$CHECKPOINT" \
    --dataset humaneval \
    --data-file "$TASKS" \
    --output-dir "$dir/shards" \
    --max-model-calls 512 \
    --transfer-tokens 1 \
    --seed-function-signature \
    --runtime-config-label "semantic-seeded-$label" \
    --initial-root-slots 1 \
    --initial-body-slots 2 \
    --initial-statement-masks "$masks" \
    --initial-function-header-masks 4 \
    --initial-loop-header-masks 4 \
    --initial-condition-masks 3 \
    --max-tree-depth 4 \
    --max-lines-per-body 16 \
    --max-total-lines 64 \
    --max-tokens-per-hole 64 \
    --max-expansions 256 \
    --body-construct-logit-penalty 4 \
    --resume
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --expected "$EXPECTED"
  "$ENV_DIR/bin/python" -m evalplus.evaluate humaneval \
    --samples "$dir/solutions.jsonl" \
    --parallel 64 \
    --test-details \
    --output-file "$dir/eval_results.json"
}

run_length stmt_4 4
run_length stmt_8 8
run_length stmt_16 16

"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import ast
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
runs = []
for label, masks in (("stmt_4", 4), ("stmt_8", 8), ("stmt_16", 16)):
    directory = root / label
    evaluation = json.load(open(directory / "eval_results.json"))
    solutions = [
        json.loads(line)
        for line in (directory / "solutions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    metrics = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    parseable = 0
    for row in solutions:
        try:
            ast.parse(str(row.get("solution", "")))
            parseable += 1
        except SyntaxError:
            pass
    reasons = Counter(
        (row.get("failure_process") or {}).get(
            "termination_reason",
            "unknown",
        )
        for row in metrics
        if row.get("error")
    )
    runs.append(
        {
            "label": label,
            "statement_masks": masks,
            "base_pass1": evaluation["pass_at_k"]["base"]["pass@1"],
            "plus_pass1": evaluation["pass_at_k"]["plus"]["pass@1"],
            "parse_rate": parseable / len(solutions),
            "errors": sum(bool(row.get("error")) for row in metrics),
            "failure_rate": (
                sum(bool(row.get("error")) for row in metrics)
                / len(metrics)
            ),
            "termination_reasons": dict(reasons),
        }
    )
eligible = [
    row
    for row in runs
    if row["failure_rate"] <= 0.05 and row["parse_rate"] >= 0.90
]
report = {
    "fixed": {
        "seed_function_signature": True,
        "body_construct_logit_penalty": 4.0,
    },
    "runs": runs,
    "selected": (
        max(
            eligible,
            key=lambda row: (
                row["plus_pass1"],
                row["parse_rate"],
                -row["statement_masks"],
            ),
        )
        if eligible
        else None
    ),
}
(root / "report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

date --iso-8601=seconds \
  >"$ROOT/ops/control/semantic_scaffold_seeded_length_calibration.done"
