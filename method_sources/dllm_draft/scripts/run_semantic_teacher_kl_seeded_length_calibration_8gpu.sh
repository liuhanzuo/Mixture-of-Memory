#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CHECKPOINT="$ROOT/outputs/semantic_teacher_kl_scale_calibration/checkpoint_scale_0125"
OUT="$ROOT/outputs/semantic_teacher_kl_seeded_length_calibration"
SUCCESS="$ROOT/ops/control/semantic_teacher_kl_seeded_length_calibration.done"
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
    --runtime-config-label "teacher-kl-seeded-$label" \
    --initial-root-slots 1 --initial-body-slots 2 \
    --initial-statement-masks "$masks" \
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
for label, masks in (
    ("stmt_4", 4),
    ("stmt_8", 8),
    ("stmt_16", 16),
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
            "statement_masks": masks,
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
    "runs": runs,
    "selected": (
        min(
            eligible,
            key=lambda row: (
                row["statement_masks"],
                -row["parseable"],
                row["errors"],
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
