#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
BASE="$ROOT/outputs/semantic_teacher_kl_scale_calibration/checkpoint_scale_0125"
ADAPTER="$ROOT/outputs/semantic_teacher_kl_leaf_elastic_pilot/global_step_64"
MERGED="$ROOT/outputs/semantic_teacher_kl_leaf_elastic_merged"
OUT="$ROOT/outputs/semantic_teacher_kl_leaf_elastic_gate"
SUCCESS="$ROOT/ops/control/semantic_teacher_kl_leaf_elastic_gate.done"
EXPECTED=16

test -s "$BASE/model.safetensors.index.json"
test -s "$ADAPTER/adapter_model.safetensors"
test ! -e "$MERGED"
test ! -e "$OUT"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

"$ENV_DIR/bin/python" "$ROOT/scripts/merge_scaffold_lora.py" \
  --base "$BASE" --adapter "$ADAPTER" --output "$MERGED"

run_vanilla() {
  local dir="$OUT/vanilla"
  mkdir -p "$dir"
  "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_dream.py" \
    --checkpoint "$MERGED" \
    --dataset humaneval \
    --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
    --output-dir "$dir/shards" \
    --limit "$EXPECTED" --steps 512 --max-new-tokens 512 \
    --temperature 0.0 --suppress-scaffold-tokens
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" --expected "$EXPECTED"
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
    --limit "$EXPECTED" --max-model-calls 512 --transfer-tokens 1 \
    --seed-function-signature \
    --runtime-config-label teacher-kl-leaf-elastic-gate \
    --initial-root-slots 1 --initial-body-slots 2 \
    --initial-statement-masks 4 \
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
    payload = {
        "tasks": len(solutions),
        "parseable": parseable,
        "nonempty": nonempty,
        "with_function": with_function,
        "errors": sum(bool(row.get("error")) for row in metrics),
    }
    if label == "scaffold":
        processes = [
            row.get("process") or row.get("failure_process") or {}
            for row in metrics
        ]
        payload.update(
            {
                "total_expansions": sum(
                    int(process.get("expansions") or 0)
                    for process in processes
                ),
                "tasks_with_expansion": sum(
                    int(process.get("expansions") or 0) > 0
                    for process in processes
                ),
                "mean_nfe": (
                    sum(int(process.get("nfe") or 0) for process in processes)
                    / len(processes)
                ),
                "maximum_tokens_per_hole": max(
                    int(process.get("maximum_tokens_per_hole") or 0)
                    for process in processes
                ),
            }
        )
    report[label] = payload

baseline_parseable = 8
report["continue"] = (
    report["vanilla"]["errors"] == 0
    and report["vanilla"]["parseable"] >= 8
    and report["scaffold"]["errors"] <= 4
    and report["scaffold"]["nonempty"] >= 12
    and report["scaffold"]["with_function"] >= 12
    and report["scaffold"]["parseable"] >= baseline_parseable
    and (
        report["scaffold"]["parseable"] > baseline_parseable
        or report["scaffold"]["tasks_with_expansion"] > 0
    )
)
(root / "report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
if not report["continue"]:
    raise SystemExit("leaf-elastic decode gate failed")
PY

printf '%s\n' "$MERGED" \
  >"$ROOT/ops/artifacts/semantic_teacher_kl_leaf_elastic_merged.txt"
date --iso-8601=seconds >"$SUCCESS"
