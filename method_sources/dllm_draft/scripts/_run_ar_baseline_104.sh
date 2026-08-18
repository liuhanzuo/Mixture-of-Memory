#!/usr/bin/env bash
# ==============================================================================
# Autoregressive (AR) EvalPlus baseline on 8x H20 (.104) -- the missing matched
# control for every diffusion number in this repo.
#
# Why Qwen2.5-Coder-7B: Dream-Coder-v0-Base-7B is adapted *from* Qwen2.5-Coder-7B
# and its config.json is architecturally identical (hidden 3584, 28 layers,
# 28 heads / 4 KV heads, intermediate 18944, vocab 152064, rope_theta 1e6).
# So this is a matched-capacity AR control, not a loose reference point.
#
# Protocol parity with the diffusion runs:
#   * prompt construction, python extraction and task sharding are imported
#     from scripts/generate_evalplus_dream.py (not reimplemented);
#   * base-LM continuation protocol (--no-chat --base-continuation), the same
#     protocol scripts/_run_dream_base_protocol_smoke_104.sh established for
#     Dream-Coder Base;
#   * sampler T=0.1 / top_p=0.95, matching the Dream-Coder Instruct baseline
#     recipe in scripts/_run_baselines_wzc1_8gpu.sh;
#   * max_new_tokens 512, matching the diffusion canvas length L=512;
#   * grading is the official evalplus.evaluate (full sets are complete, so no
#     coverage-assertion workaround is needed here).
#
# Sharding: per-shard CUDA_VISIBLE_DEVICES=$g, so torch sees exactly one GPU
# and LOCAL_RANK MUST be 0; RANK=$g is the logical shard id used for
# `index % world_size == rank`. Setting LOCAL_RANK=$g here would make shards
# 1..7 die with "invalid device ordinal".
# ==============================================================================
set -uo pipefail

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104
PY="$ROOT/.venv_dream/bin/python"
CKPT="$ROOT/models/Qwen2.5-Coder-7B"
# Defaults reproduce the diffusion-matched sampler (T=0.1 / top_p=0.95). Set
# AR_TEMP=0 AR_TOPP=1.0 AR_RUN_NAME=..._greedy for the deterministic arm that
# matches the Qwen2.5-Coder technical report's own decoding.
RUN_NAME="${AR_RUN_NAME:-ar_qwen25coder7b_base}"
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
export MBPP_OVERRIDE_PATH="$ROOT/data/evalplus/MbppPlus-v0.2.0.jsonl"

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

  # Hard coverage assertion: a silently missing shard would corrupt pass@1.
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

  # Official EvalPlus grader. No hand-written test runner anywhere.
  "$PY" -m evalplus.evaluate "$dataset" \
    --samples "$dir/solutions.jsonl" \
    --parallel 32 \
    --test-details \
    --output-file "$dir/eval_results.json" || return 1

  "$PY" "$ROOT/scripts/_summarize_ar_run.py" \
    --dataset "$dataset" \
    --metrics "$dir/metrics.jsonl" \
    --eval-results "$dir/eval_results.json" \
    --output "$dir/report.json" || return 1
}

rc=0
run_dataset humaneval "$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" 164 || rc=1
run_dataset mbpp      "$ROOT/data/evalplus/MbppPlus-v0.2.0.jsonl"       378 || rc=1

if [ "$rc" -eq 0 ]; then
  date --iso-8601=seconds > "$SUCCESS"
  echo "[$(date '+%F %T')] ===== AR BASELINE DONE ====="
else
  echo "[$(date '+%F %T')] ===== AR BASELINE FAILED (rc=$rc) =====" >&2
fi
exit "$rc"
