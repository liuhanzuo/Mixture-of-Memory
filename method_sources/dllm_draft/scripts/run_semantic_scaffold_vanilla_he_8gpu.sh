#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CHECKPOINT_FILE="${SEMANTIC_CHECKPOINT_FILE:-$ROOT/ops/artifacts/semantic_scaffold_1ep_merged_checkpoint.txt}"
OUT="${SEMANTIC_VANILLA_OUT:-$ROOT/outputs/semantic_scaffold_vanilla_humaneval}"
SUCCESS="${SEMANTIC_VANILLA_SUCCESS:-$ROOT/ops/control/semantic_scaffold_vanilla_humaneval.done}"

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
  "$ROOT/scripts/generate_evalplus_dream.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset humaneval \
  --data-file "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  --output-dir "$OUT/shards" \
  --steps 512 \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --suppress-scaffold-tokens \
  --resume

"$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
  --input-dir "$OUT/shards" \
  --solutions "$OUT/solutions.jsonl" \
  --metrics "$OUT/metrics.jsonl" \
  --expected 164
"$ENV_DIR/bin/python" "$ROOT/scripts/reextract_dream_solutions.py" \
  --metrics "$OUT/metrics.jsonl" \
  --solutions "$OUT/solutions.jsonl"
"$ENV_DIR/bin/python" -m evalplus.evaluate humaneval \
  --samples "$OUT/solutions.jsonl" \
  --parallel 64 \
  --test-details \
  --output-file "$OUT/eval_results.json"

date --iso-8601=seconds >"$SUCCESS"
