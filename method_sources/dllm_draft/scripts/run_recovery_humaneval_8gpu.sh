#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
ARM="${RECOVERY_ARM:?set RECOVERY_ARM}"

case "$ARM" in
  base_raw)
    CHECKPOINT="$ROOT/models/Dream-Coder-v0-Base-7B"
    OUT="$ROOT/outputs/recovery_base_raw_humaneval"
    SUCCESS="$ROOT/ops/control/recovery_base_raw_humaneval.done"
    ;;
  base_plain_1ep)
    CHECKPOINT="$(cat "$ROOT/ops/artifacts/recovery_base_plain_1ep_checkpoint.txt")"
    OUT="$ROOT/outputs/recovery_base_plain_1ep_humaneval"
    SUCCESS="$ROOT/ops/control/recovery_base_plain_1ep_humaneval.done"
    ;;
  instruct_plain_1ep)
    CHECKPOINT="$(cat "$ROOT/ops/artifacts/recovery_instruct_plain_1ep_checkpoint.txt")"
    OUT="$ROOT/outputs/recovery_instruct_plain_1ep_humaneval"
    SUCCESS="$ROOT/ops/control/recovery_instruct_plain_1ep_humaneval.done"
    ;;
  instruct_highnoise_1ep)
    CHECKPOINT="$(cat "$ROOT/ops/artifacts/recovery_instruct_highnoise_1ep_checkpoint.txt")"
    OUT="$ROOT/outputs/recovery_instruct_highnoise_1ep_humaneval"
    SUCCESS="$ROOT/ops/control/recovery_instruct_highnoise_1ep_humaneval.done"
    ;;
  *)
    echo "Unsupported RECOVERY_ARM=$ARM" >&2
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
  --steps 512 \
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
