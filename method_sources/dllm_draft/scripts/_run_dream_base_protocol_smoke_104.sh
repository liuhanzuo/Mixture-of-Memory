#!/usr/bin/env bash
# Verify the official Dream-Coder Base prompt protocol before a full rerun.
# The previous 8.5% HE result used an instruction wrapper and scored only the
# generated suffix. The paper's lm-eval path instead prepends BOS, feeds the
# raw function-prefix prompt, and evaluates prompt + continuation.
set -euo pipefail

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104
PY="$ROOT/.venv_dream/bin/python"
OUT="$ROOT/outputs/dream_base_protocol_he16"
SUCCESS="$ROOT/ops/control/dream_base_protocol_he16.done"

cd "$ROOT"
test -s models/Dream-Coder-v0-Base-7B/model.safetensors.index.json
test ! -e "$OUT"
mkdir -p "$OUT/shards"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"

"$ROOT/.venv_dream/bin/torchrun" --standalone --nproc_per_node=8 \
  scripts/generate_evalplus_dream.py \
  --checkpoint models/Dream-Coder-v0-Base-7B \
  --dataset humaneval \
  --data-file data/evalplus/HumanEvalPlus-v0.1.10.jsonl \
  --output-dir "$OUT/shards" \
  --limit 16 \
  --steps 512 \
  --max-new-tokens 512 \
  --temperature 0.2 \
  --top-p 0.9 \
  --no-chat \
  --add-bos-token \
  --base-continuation

"$PY" scripts/merge_evalplus_shards.py \
  --input-dir "$OUT/shards" \
  --solutions "$OUT/solutions.jsonl" \
  --metrics "$OUT/metrics.jsonl" \
  --expected 16
"$PY" -m evalplus.evaluate humaneval \
  --samples "$OUT/solutions.jsonl" \
  --parallel 16 \
  --test-details \
  --output-file "$OUT/eval_results.json"

"$PY" - "$OUT" <<'PY'
import ast
import json
import sys
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
evaluation = json.loads((root / "eval_results.json").read_text())
parseable = 0
for row in solutions:
    try:
        ast.parse(str(row.get("solution", "")))
        parseable += 1
    except SyntaxError:
        pass
report = {
    "tasks": len(solutions),
    "parseable": parseable,
    "errors": sum(bool(row.get("error")) for row in metrics),
    "base_pass1": evaluation["pass_at_k"]["base"]["pass@1"],
    "plus_pass1": evaluation["pass_at_k"]["plus"]["pass@1"],
    "protocol": {
        "raw_benchmark_prefix": True,
        "add_bos_token": True,
        "prompt_plus_continuation": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "steps": 512,
    },
}
(root / "report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
if report["errors"] != 0 or report["parseable"] < 8:
    raise SystemExit("Dream-Coder Base protocol smoke failed")
PY

date --iso-8601=seconds >"$SUCCESS"
