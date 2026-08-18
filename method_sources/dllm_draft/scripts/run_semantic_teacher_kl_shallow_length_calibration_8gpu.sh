#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CHECKPOINT="$ROOT/outputs/semantic_teacher_kl_scale_calibration/checkpoint_scale_0125"
OUT="$ROOT/outputs/semantic_teacher_kl_shallow_length_calibration"
SUCCESS="$ROOT/ops/control/semantic_teacher_kl_shallow_length_calibration.done"
EXPECTED=16

test -s "$CHECKPOINT/model.safetensors.index.json"
test ! -e "$OUT"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

run_length() {
  local label="$1" masks="$2"
  local dir="$OUT/$label"
  "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_scaffold.py" \
    --checkpoint "$CHECKPOINT" \
    --dataset humaneval \
    --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
    --output-dir "$dir/shards" \
    --limit "$EXPECTED" --max-model-calls 512 --transfer-tokens 1 \
    --seed-function-signature \
    --runtime-config-label "teacher-kl-shallow-$label" \
    --initial-root-slots 1 --initial-body-slots 2 \
    --initial-statement-masks 4 \
    --initial-statement-masks-shallow "$masks" \
    --statement-shallow-depth 1 \
    --initial-function-header-masks 4 \
    --initial-loop-header-masks 4 --initial-condition-masks 3 \
    --max-tree-depth 4 --max-lines-per-body 16 \
    --max-total-lines 64 --max-tokens-per-hole 64 \
    --max-expansions 256 \
    --body-stmt-logit-bonus 4
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" --expected "$EXPECTED"
}

run_length shallow_6 6
run_length shallow_8 8
run_length shallow_12 12

"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import ast
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
runs = []
for label, masks in (
    ("shallow_6", 6),
    ("shallow_8", 8),
    ("shallow_12", 12),
):
    directory = root / label
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
    parseable = nonempty = with_function = 0
    for row in solutions:
        source = str(row.get("solution", ""))
        nonempty += bool(source.strip())
        with_function += "def " in source
        try:
            ast.parse(source)
            parseable += 1
        except SyntaxError:
            pass
    reasons = Counter(
        (row.get("failure_process") or {}).get(
            "termination_reason", "unknown"
        )
        for row in metrics
        if row.get("error")
    )
    runs.append(
        {
            "label": label,
            "shallow_statement_masks": masks,
            "errors": sum(bool(row.get("error")) for row in metrics),
            "nonempty": nonempty,
            "with_function": with_function,
            "parseable": parseable,
            "termination_reasons": dict(reasons),
        }
    )
eligible = [
    row
    for row in runs
    if row["errors"] <= 4
    and row["nonempty"] >= 12
    and row["with_function"] >= 12
    and row["parseable"] >= 8
]
report = {
    "checkpoint_scale": 0.125,
    "body_stmt_logit_bonus": 4.0,
    "seed_function_signature": True,
    "nested_statement_masks": 4,
    "runs": runs,
    "selected": (
        max(
            eligible,
            key=lambda row: (
                row["parseable"],
                -row["shallow_statement_masks"],
                -row["errors"],
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

date --iso-8601=seconds >"$SUCCESS"
