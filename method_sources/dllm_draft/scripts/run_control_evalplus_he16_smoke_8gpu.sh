#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
MODE="${CONTROL_MODE:?set CONTROL_MODE=schedule_only or plain}"

case "$MODE" in
  schedule_only)
    CHECKPOINT_FILE="$ROOT/ops/artifacts/schedule_only_stage1_latest_checkpoint.txt"
    OUT="$ROOT/outputs/schedule_only_evalplus_he16_smoke"
    SUCCESS="$ROOT/ops/control/schedule_only_evalplus_he16_smoke.done"
    ;;
  plain)
    CHECKPOINT_FILE="$ROOT/ops/artifacts/plain_stage1_latest_checkpoint.txt"
    OUT="$ROOT/outputs/plain_evalplus_he16_smoke"
    SUCCESS="$ROOT/ops/control/plain_evalplus_he16_smoke.done"
    ;;
  *)
    echo "Unsupported CONTROL_MODE=$MODE" >&2
    exit 2
    ;;
esac

test -s "$CHECKPOINT_FILE"
CHECKPOINT="$(cat "$CHECKPOINT_FILE")"
test -d "$CHECKPOINT"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"

"$ENV_DIR/bin/torchrun" \
  --standalone \
  --nproc_per_node=8 \
  "$ROOT/scripts/generate_evalplus_dream.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset humaneval \
  --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  --output-dir "$OUT/shards" \
  --limit 16 \
  --steps 128 \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --resume

"$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
  --input-dir "$OUT/shards" \
  --solutions "$OUT/solutions.jsonl" \
  --metrics "$OUT/metrics.jsonl" \
  --expected 16
"$ENV_DIR/bin/python" "$ROOT/scripts/reextract_dream_solutions.py" \
  --metrics "$OUT/metrics.jsonl" \
  --solutions "$OUT/solutions.jsonl"

"$ENV_DIR/bin/python" - "$MODE" "$OUT/solutions.jsonl" "$OUT/metrics.jsonl" "$OUT/summary.json" <<'PY'
import ast
import json
import sys
from pathlib import Path

mode = sys.argv[1]
solutions = [
    json.loads(line)
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
metrics = [
    json.loads(line)
    for line in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
parseable = 0
nonempty = 0
for row in solutions:
    source = str(row.get("solution", ""))
    if not source.strip():
        continue
    nonempty += 1
    try:
        ast.parse(source)
        parseable += 1
    except SyntaxError:
        pass
summary = {
    "mode": mode,
    "rows": len(solutions),
    "generation_failures": sum(bool(row.get("error")) for row in metrics),
    "nonempty": nonempty,
    "parseable": parseable,
}
Path(sys.argv[4]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(summary)
if summary["rows"] != 16:
    raise SystemExit("control smoke coverage mismatch")
if summary["generation_failures"]:
    raise SystemExit("control smoke had generation failures")
if summary["nonempty"] == 0 or summary["parseable"] == 0:
    raise SystemExit("control smoke produced no usable Python")
PY

date --iso-8601=seconds >"$SUCCESS"
