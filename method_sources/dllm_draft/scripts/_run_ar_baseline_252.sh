#!/usr/bin/env bash
# ==============================================================================
# AR baseline on 8x .252 (L20A, wzc1) — matched control to .104 AR baseline.
#
# This is the "highest-value follow-up" flagged in CROSSNODE_REPRODUCIBILITY.md
# §4 limit #2: does an AR model (Qwen2.5-Coder-7B) show the same 2.44 pt HE+
# gap across GPU architectures (H20 vs L20A)? If YES, the "dLLMs are unusually
# exposed" framing dies and this becomes a generic bf16-reproducibility fact.
# If NO (0 flips as expected under greedy AR with typical stacks), the dLLM
# framing survives.
#
# Adapted from scripts/_run_ar_baseline_104.sh: path substitution to wzc1 root,
# python swapped to .venv_b200, evalplus/torch/tf versions match arm A/B of the
# crossnode audit (0.3.1 / 2.11.0+cu128 / 4.51.3) so grader is common with the
# dLLM headline.
# ==============================================================================
set -uo pipefail

ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY="$ROOT/.venv_b200/bin/python"
CKPT="$ROOT/models/Qwen2.5-Coder-7B"
RUN_NAME="${AR_RUN_NAME:-ar_qwen25coder7b_base_252}"
OUT="$ROOT/outputs/$RUN_NAME"
SUCCESS="$ROOT/ops/control/$RUN_NAME.done"
NG=8
MAXNEW="${AR_MAXNEW:-512}"
TEMP="${AR_TEMP:-0.1}"
TOPP="${AR_TOPP:-0.95}"

cd "$ROOT" || exit 1
mkdir -p "$OUT" logs "$ROOT/ops/control"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"

run_dataset () {
  local dataset="$1" data_file="$2" expected="$3"
  local dir="$OUT/$dataset"

  if [ -f "$dir/eval_results.json" ]; then
    echo "[$(date '+%F %T')] $dataset: eval_results.json exists -> SKIP"
    return 0
  fi
  mkdir -p "$dir/shards"
  echo "[$(date '+%F %T')] ===== AR $dataset (n=$expected, ${NG} shards) ====="

  for g in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=$NG \
      "$PY" -u "$ROOT/scripts/generate_evalplus_ar.py" \
        --checkpoint "$CKPT" \
        --dataset "$dataset" \
        --data-file "$data_file" \
        --output-dir "$dir/shards" \
        --max-new-tokens "$MAXNEW" \
        --temperature "$TEMP" \
        --top-p "$TOPP" \
        --no-chat \
        --base-continuation \
        --resume \
        > "$dir/shard${g}.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] $dataset: all shards returned"

  local n_shards
  n_shards=$(ls "$dir"/shards/solutions.rank*.jsonl 2>/dev/null | wc -l)
  if [ "$n_shards" -ne "$NG" ]; then
    echo "FATAL $dataset: only $n_shards/$NG shard files present" >&2
    grep -l "Traceback" "$dir"/shard*.log 2>/dev/null >&2
    return 1
  fi

  "$PY" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$dir/shards" \
    --solutions "$dir/solutions.jsonl" \
    --metrics "$dir/metrics.jsonl" \
    --expected "$expected" || return 1

  "$PY" -m evalplus.evaluate "$dataset" \
    --samples "$dir/solutions.jsonl" \
    --parallel 32 \
    --test-details \
    --output-file "$dir/eval_results.json" || return 1

  if [ -x "$ROOT/scripts/_summarize_ar_run.py" ] || [ -f "$ROOT/scripts/_summarize_ar_run.py" ]; then
    "$PY" "$ROOT/scripts/_summarize_ar_run.py" \
      --dataset "$dataset" \
      --metrics "$dir/metrics.jsonl" \
      --eval-results "$dir/eval_results.json" \
      --output "$dir/report.json" || true
  fi
}

rc=0
run_dataset humaneval "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" 164 || rc=1

if [ "$rc" -eq 0 ]; then
  date --iso-8601=seconds > "$SUCCESS"
  echo "[$(date '+%F %T')] ===== AR BASELINE ON .252 DONE ====="
else
  echo "[$(date '+%F %T')] ===== AR BASELINE ON .252 FAILED (rc=$rc) =====" >&2
fi
exit "$rc"
