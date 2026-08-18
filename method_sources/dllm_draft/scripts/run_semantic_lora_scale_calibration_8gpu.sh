#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
ADAPTER="$(cat "$ROOT/ops/artifacts/semantic_scaffold_lora_1ep_checkpoint.txt")"
BASE="$ROOT/models/Dream-Coder-v0-Instruct-7B"
OUT="$ROOT/outputs/semantic_lora_scale_calibration"
TASKS="$ROOT/outputs/runtime_capacity_screen/tasks.jsonl"
EXPECTED=64

test -s "$ADAPTER/adapter_model.safetensors"
test -s "$TASKS"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# EvalPlus must load the same 64-task problem set as generation; otherwise it
# asserts that the remaining 100 full-benchmark tasks are missing.
export HUMANEVAL_OVERRIDE_PATH="$TASKS"

run_scale() {
  local label="$1" scale="$2"
  local checkpoint="$OUT/checkpoint_$label"
  local dir="$OUT/$label"
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_scaffold_lora.py" \
    --base "$BASE" --adapter "$ADAPTER" --output "$checkpoint" \
    --scale "$scale"
  "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_dream.py" \
    --checkpoint "$checkpoint" --dataset humaneval \
    --data-file "$TASKS" --output-dir "$dir/shards" \
    --steps 512 --max-new-tokens 512 --temperature 0.0 \
    --suppress-scaffold-tokens --resume
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" --expected "$EXPECTED"
  "$ENV_DIR/bin/python" "$ROOT/scripts/reextract_dream_solutions.py" \
    --metrics "$dir/metrics.jsonl" --solutions "$dir/solutions.jsonl"
  "$ENV_DIR/bin/python" -m evalplus.evaluate humaneval \
    --samples "$dir/solutions.jsonl" --parallel 64 --test-details \
    --output-file "$dir/eval_results.json"
}

run_scale scale_025 0.25
run_scale scale_050 0.50
run_scale scale_075 0.75

"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import ast, json, sys
from pathlib import Path

root=Path(sys.argv[1])
rows=[]
for label,scale in (("scale_025",0.25),("scale_050",0.50),("scale_075",0.75)):
 d=root/label
 ev=json.load(open(d/"eval_results.json"))
 sols=[json.loads(x) for x in (d/"solutions.jsonl").read_text().splitlines() if x.strip()]
 metrics=[json.loads(x) for x in (d/"metrics.jsonl").read_text().splitlines() if x.strip()]
 parse=0
 for row in sols:
  try: ast.parse(str(row.get("solution",""))); parse+=1
  except SyntaxError: pass
 rows.append({
  "label":label,"scale":scale,
  "base_pass1":ev["pass_at_k"]["base"]["pass@1"],
  "plus_pass1":ev["pass_at_k"]["plus"]["pass@1"],
  "parse_rate":parse/len(sols),
  "errors":sum(bool(x.get("error")) for x in metrics),
 })
report={"runs":rows,"selected":max(rows,key=lambda x:(x["plus_pass1"],x["parse_rate"]))}
(root/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
PY

date --iso-8601=seconds \
  >"$ROOT/ops/control/semantic_lora_scale_calibration.done"
