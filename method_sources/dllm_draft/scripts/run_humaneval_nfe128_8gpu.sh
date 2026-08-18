#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
MODE="${CONTROL_MODE:?set CONTROL_MODE=dream, schedule_only, or plain}"

case "$MODE" in
  dream)
    CHECKPOINT="$ROOT/models/Dream-Coder-v0-Instruct-7B"
    OUT="$ROOT/outputs/dream_coder_humaneval_nfe128"
    SUCCESS="$ROOT/ops/control/dream_coder_humaneval_nfe128.done"
    ;;
  schedule_only)
    CHECKPOINT="$(cat "$ROOT/ops/artifacts/schedule_only_stage1_latest_checkpoint.txt")"
    OUT="$ROOT/outputs/schedule_only_humaneval_nfe128"
    SUCCESS="$ROOT/ops/control/schedule_only_humaneval_nfe128.done"
    ;;
  plain)
    CHECKPOINT="$(cat "$ROOT/ops/artifacts/plain_stage1_latest_checkpoint.txt")"
    OUT="$ROOT/outputs/plain_humaneval_nfe128"
    SUCCESS="$ROOT/ops/control/plain_humaneval_nfe128.done"
    ;;
  *)
    echo "Unsupported CONTROL_MODE=$MODE" >&2
    exit 2
    ;;
esac

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
  --steps 128 \
  --max-new-tokens 512 \
  --temperature 0.0 \
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
