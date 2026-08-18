#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CHECKPOINT_FILE="${SEMANTIC_CHECKPOINT_FILE:-$ROOT/ops/artifacts/semantic_scaffold_1ep_merged_checkpoint.txt}"
OUT="${SEMANTIC_SCAFFOLD_OUT:-$ROOT/outputs/semantic_scaffold_medium_humaneval}"
SUCCESS="${SEMANTIC_SCAFFOLD_SUCCESS:-$ROOT/ops/control/semantic_scaffold_medium_humaneval.done}"

test -s "$CHECKPOINT_FILE"
CHECKPOINT="$(cat "$CHECKPOINT_FILE")"
test -s "$CHECKPOINT/model.safetensors.index.json"
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
  "$ROOT/scripts/generate_evalplus_scaffold.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset humaneval \
  --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  --output-dir "$OUT/shards" \
  --max-model-calls 512 \
  --transfer-tokens 1 \
  --runtime-config-label semantic-medium \
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
  --max-expansions 256 \
  --resume

"$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
  --input-dir "$OUT/shards" \
  --solutions "$OUT/solutions.jsonl" \
  --metrics "$OUT/metrics.jsonl" \
  --expected 164
"$ENV_DIR/bin/python" -m evalplus.evaluate humaneval \
  --samples "$OUT/solutions.jsonl" \
  --parallel 64 \
  --test-details \
  --output-file "$OUT/eval_results.json"

date --iso-8601=seconds >"$SUCCESS"
