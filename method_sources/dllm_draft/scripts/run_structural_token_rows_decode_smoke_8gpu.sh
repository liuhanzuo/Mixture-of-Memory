#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CHECKPOINT_FILE="${TOKEN_ROW_CHECKPOINT_FILE:-$ROOT/ops/artifacts/structural_token_rows_native_merged_checkpoint.txt}"
CHECKPOINT="$(cat "$CHECKPOINT_FILE")"
OUT="${TOKEN_ROW_DECODE_OUT:-$ROOT/outputs/structural_token_rows_decode_smoke}"
SUCCESS="${TOKEN_ROW_DECODE_SUCCESS:-$ROOT/ops/control/structural_token_rows_decode_smoke.done}"

test -s "$CHECKPOINT/model.safetensors.index.json"
test ! -e "$OUT"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"

"$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
  "$ROOT/scripts/generate_evalplus_scaffold.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset humaneval \
  --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  --output-dir "$OUT/shards" \
  --limit 16 \
  --max-model-calls 512 \
  --transfer-tokens 1 \
  --runtime-config-label token-rows-smoke \
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

"$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
  --input-dir "$OUT/shards" \
  --solutions "$OUT/solutions.jsonl" \
  --metrics "$OUT/metrics.jsonl" \
  --expected 16
"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import ast
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
solutions = [
    json.loads(line)
    for line in (root / "solutions.jsonl").read_text().splitlines()
    if line.strip()
]
metrics = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
parseable = 0
nonempty = 0
with_function = 0
for row in solutions:
    source = str(row.get("solution", ""))
    if source.strip():
        nonempty += 1
    if "def " in source:
        with_function += 1
    try:
        ast.parse(source)
        parseable += 1
    except SyntaxError:
        pass
reasons = Counter(
    (row.get("failure_process") or {}).get("termination_reason", "unknown")
    for row in metrics
    if row.get("error")
)
summary = {
    "tasks": len(solutions),
    "generation_errors": sum(bool(row.get("error")) for row in metrics),
    "parseable": parseable,
    "nonempty": nonempty,
    "with_function": with_function,
    "termination_reasons": dict(reasons),
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["generation_errors"] > 8:
    raise SystemExit("token-row smoke did not reduce generation failures")
if summary["parseable"] < 8:
    raise SystemExit("token-row smoke parseability gate failed")
if summary["nonempty"] < 16 or summary["with_function"] < 16:
    raise SystemExit("token-row smoke collapsed to empty/non-function output")
PY

date --iso-8601=seconds \
  >"$SUCCESS"
