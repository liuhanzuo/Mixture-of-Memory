#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
ADAPTER="${SEMANTIC_TEACHER_KL_ADAPTER:-$ROOT/outputs/semantic_teacher_kl_replicated_smoke_v2/global_step_4}"
MERGED="${SEMANTIC_TEACHER_KL_MERGED:-$ROOT/outputs/semantic_teacher_kl_replicated_merged}"
OUT="${SEMANTIC_TEACHER_KL_GATE_OUT:-$ROOT/outputs/semantic_teacher_kl_decode_gate}"
SUCCESS="${SEMANTIC_TEACHER_KL_GATE_SUCCESS:-$ROOT/ops/control/semantic_teacher_kl_decode_gate.done}"

test -s "$ADAPTER/adapter_model.safetensors"
test ! -e "$MERGED"
test ! -e "$OUT"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

"$ENV_DIR/bin/python" "$ROOT/scripts/merge_scaffold_lora.py" \
  --base "$ROOT/models/Dream-Coder-v0-Instruct-7B" \
  --adapter "$ADAPTER" \
  --output "$MERGED"

run_vanilla() {
  local dir="$OUT/vanilla"
  mkdir -p "$dir"
  "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_dream.py" \
    --checkpoint "$MERGED" \
    --dataset humaneval \
    --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
    --output-dir "$dir/shards" \
    --limit 16 --steps 512 --max-new-tokens 512 \
    --temperature 0.0 --suppress-scaffold-tokens
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" --expected 16
  "$ENV_DIR/bin/python" "$ROOT/scripts/reextract_dream_solutions.py" \
    --metrics "$dir/metrics.jsonl" --solutions "$dir/solutions.jsonl"
}

run_scaffold() {
  local dir="$OUT/scaffold"
  mkdir -p "$dir"
  "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_scaffold.py" \
    --checkpoint "$MERGED" \
    --dataset humaneval \
    --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
    --output-dir "$dir/shards" \
    --limit 16 --max-model-calls 512 --transfer-tokens 1 \
    --runtime-config-label semantic-teacher-kl-gate \
    --initial-root-slots 1 --initial-body-slots 2 \
    --initial-statement-masks 4 --initial-function-header-masks 4 \
    --initial-loop-header-masks 4 --initial-condition-masks 3 \
    --max-tree-depth 4 --max-lines-per-body 16 \
    --max-total-lines 64 --max-tokens-per-hole 32 \
    --max-expansions 256
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" --expected 16
}

run_vanilla
run_scaffold

"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import ast
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = {}
for label in ("vanilla", "scaffold"):
    solutions = [
        json.loads(line)
        for line in (root / label / "solutions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    metrics = [
        json.loads(line)
        for line in (root / label / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    parseable = 0
    nonempty = 0
    with_function = 0
    for row in solutions:
        source = str(row.get("solution", ""))
        nonempty += bool(source.strip())
        with_function += "def " in source
        try:
            ast.parse(source)
            parseable += 1
        except SyntaxError:
            pass
    report[label] = {
        "tasks": len(solutions),
        "parseable": parseable,
        "nonempty": nonempty,
        "with_function": with_function,
        "errors": sum(bool(row.get("error")) for row in metrics),
    }
report["continue"] = (
    report["vanilla"]["parseable"] >= 8
    and report["vanilla"]["errors"] == 0
    and report["scaffold"]["nonempty"] >= 12
    and report["scaffold"]["with_function"] >= 12
    and report["scaffold"]["parseable"] >= 8
    and report["scaffold"]["errors"] <= 4
)
(root / "report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
if not report["continue"]:
    raise SystemExit("teacher-KL dual decode gate failed")
PY

printf '%s\n' "$MERGED" \
  >"$ROOT/ops/artifacts/semantic_teacher_kl_merged_checkpoint.txt"
date --iso-8601=seconds >"$SUCCESS"
