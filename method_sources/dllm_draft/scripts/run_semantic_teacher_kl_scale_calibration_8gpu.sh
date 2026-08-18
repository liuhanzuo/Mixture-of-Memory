#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
ADAPTER="$ROOT/outputs/semantic_teacher_kl_replicated_smoke_v2/global_step_4"
OUT="$ROOT/outputs/semantic_teacher_kl_scale_calibration"
SUCCESS="$ROOT/ops/control/semantic_teacher_kl_scale_calibration.done"
EXPECTED=16

test -s "$ADAPTER/adapter_model.safetensors"
test ! -e "$OUT"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

run_scale() {
  local label="$1" scale="$2"
  local checkpoint="$OUT/checkpoint_$label"
  local dir="$OUT/$label"
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_scaffold_lora.py" \
    --base "$ROOT/models/Dream-Coder-v0-Instruct-7B" \
    --adapter "$ADAPTER" --output "$checkpoint" --scale "$scale"
  "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_scaffold.py" \
    --checkpoint "$checkpoint" \
    --dataset humaneval \
    --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
    --output-dir "$dir/shards" \
    --limit "$EXPECTED" --max-model-calls 512 --transfer-tokens 1 \
    --runtime-config-label "teacher-kl-$label" \
    --initial-root-slots 1 --initial-body-slots 2 \
    --initial-statement-masks 4 --initial-function-header-masks 4 \
    --initial-loop-header-masks 4 --initial-condition-masks 3 \
    --max-tree-depth 4 --max-lines-per-body 16 \
    --max-total-lines 64 --max-tokens-per-hole 32 \
    --max-expansions 256
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" --expected "$EXPECTED"
}

run_scale scale_0125 0.125
run_scale scale_0250 0.250
run_scale scale_0500 0.500

"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import ast
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
runs = []
for label, scale in (
    ("scale_0125", 0.125),
    ("scale_0250", 0.250),
    ("scale_0500", 0.500),
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
            "scale": scale,
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
    "runs": runs,
    "selected": (
        max(
            eligible,
            key=lambda row: (
                row["scale"],
                row["parseable"],
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
