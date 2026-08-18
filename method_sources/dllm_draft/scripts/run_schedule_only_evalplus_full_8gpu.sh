#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
OUT="$ROOT/outputs/schedule_only_evalplus_full"
CHECKPOINT="$(cat "$ROOT/ops/artifacts/schedule_only_stage1_latest_checkpoint.txt")"

export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"
export MBPP_OVERRIDE_PATH="$ROOT/data/evalplus/MbppPlus-v0.2.0.jsonl"

run_dataset() {
  local dataset="$1" data_file="$2" expected="$3"
  local dir="$OUT/$dataset"
  mkdir -p "$dir"
  "$ENV_DIR/bin/torchrun" \
    --standalone \
    --nproc_per_node=8 \
    "$ROOT/scripts/generate_evalplus_dream.py" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$dataset" \
    --data-file "$data_file" \
    --output-dir "$dir/shards" \
    --steps 512 \
    --max-new-tokens 512 \
    --temperature 0.0 \
    --resume
  "$ENV_DIR/bin/python" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --expected "$expected"
  "$ENV_DIR/bin/python" "$ROOT/scripts/reextract_dream_solutions.py" \
    --metrics "$dir/metrics.jsonl" \
    --solutions "$dir/solutions.jsonl"
  "$ENV_DIR/bin/python" -m evalplus.evaluate "$dataset" \
    --samples "$dir/solutions.jsonl" \
    --parallel 64 \
    --test-details \
    --output-file "$dir/eval_results.json"
}

run_dataset \
  humaneval \
  "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" \
  164
run_dataset \
  mbpp \
  "$ROOT/data/evalplus/MbppPlus-v0.2.0.jsonl" \
  378

date --iso-8601=seconds \
  >"$ROOT/ops/control/schedule_only_evalplus_full.done"
