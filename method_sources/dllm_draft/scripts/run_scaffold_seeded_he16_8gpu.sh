#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
OUT="$ROOT/outputs/evalplus_humaneval_seeded_smoke"
CHECKPOINT="$(cat "$ROOT/ops/artifacts/scaffold_stage1_latest_checkpoint.txt")"

mkdir -p "$OUT"
export PYTHONPATH="$ROOT:$ROOT/vendor/DreamOn:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

"$ENV_DIR/bin/torchrun" \
  --standalone \
  --nproc_per_node=8 \
  "$ROOT/scripts/generate_evalplus_scaffold.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset humaneval \
  --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  --output-dir "$OUT/shards" \
  --limit 16 \
  --max-model-calls 512 \
  --seed-function-signature

"$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
  --input-dir "$OUT/shards" \
  --solutions "$OUT/solutions.jsonl" \
  --metrics "$OUT/metrics.jsonl" \
  --expected 16

"$ENV_DIR/bin/python" - "$OUT/metrics.jsonl" "$OUT/summary.json" <<'PY'
import json, sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
summary={
    "rows":len(rows),
    "failures":sum(bool(x["error"]) for x in rows),
    "parseable":sum(bool(x["process"] and x["process"]["final_parseable"]) for x in rows),
}
open(sys.argv[2],"w").write(json.dumps(summary,indent=2)+"\n")
print(summary)
if summary["failures"]:
    raise SystemExit("seeded Scaffold generation failures")
PY

date --iso-8601=seconds \
  >"$ROOT/ops/control/scaffold_seeded_he16.done"
