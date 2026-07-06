#!/usr/bin/env bash
# ============================================================================
# QCMem Direction-B ablation — non-contiguous "top-prepay" resume b-sweep
# (2026-07-05). Extends the 2-group task-pool scheduler to sweep top_prepay_b
# at a FIXED resume_j (default 12). A "task" = (b, babilong-task, length).
#
# Direction B hypothesis: only the MIDDLE integration band needs query-aware
# recompute; the TOP b layers are query-blind "output/format" layers and can be
# run query-LOCAL (over the query tail only, not the full context). This saves
# running the top b layers over the long context. b=0 == exact connective
# resume (the current QCMem read). See versions/v_qcmem_top_prepay.md for the
# feasibility analysis (exact top-prepay is impossible; this is the tractable
# approximation, whose quality is the ablation question).
#
# Zero-training (plain Qwen) OR distilled (--lora_adapter) — pass LORA_ADAPTER.
#
# Usage:
#   RESUME_J=12 B_VALUES="0 4 8 12" \
#   MODEL=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
#   [LORA_ADAPTER=outputs/qcmem_distill_qwen_j12b0_pg19_nctx7/final] \
#   [TASKS="qa1 qa2 qa5"] [LENGTHS="0k 1k 2k 4k 8k 16k"] \
#   PROJECT_ROOT=<node root> PYTHON_BIN=<node .venv> \
#   setsid nohup bash scripts/_eval_qcmem_bsweep_taskpool.sh \
#     >logs/qcmem_bsweep.out 2>&1 &
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="" https_proxy="" all_proxy=""
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_cache/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

MODEL="${MODEL:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b}"
RESUME_J="${RESUME_J:-12}"
B_VALUES="${B_VALUES:-0 4 8 12}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
SELECTOR="${SELECTOR:-bm25}"
TOPK="${TOPK:-4}"
SINK="${SINK:-bos}"
TASKS="${TASKS:-qa1 qa2 qa5}"
LENGTHS="${LENGTHS:-0k 1k 2k 4k 8k 16k}"
NSHARD="${NSHARD:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
SUFFIX_BASE="_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no"
RUN_TAG="${RUN_TAG:-}"   # optional suffix to distinguish distilled vs zero-train

read -r -a B_ARR    <<< "$B_VALUES"
read -r -a TASK_ARR <<< "$TASKS"
read -r -a LEN_ARR  <<< "$LENGTHS"
read -r -a GROUP0 <<< "${GROUP0_GPUS:-0 1 2 3}"
read -r -a GROUP1 <<< "${GROUP1_GPUS:-4 5 6 7}"
NUM_GROUPS="${NUM_GROUPS:-2}"

LORA_ARG=(); [ -n "$LORA_ADAPTER" ] && LORA_ARG=(--lora_adapter "$LORA_ADAPTER")

LOGDIR="logs/eval_qcmem_bsweep_j${RESUME_J}${RUN_TAG}"
mkdir -p "$LOGDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

: > "$QUEUE"
for b in "${B_ARR[@]}"; do
  for task in "${TASK_ARR[@]}"; do
    for L in "${LEN_ARR[@]}"; do
      echo "$b $task $L" >> "$QUEUE"
    done
  done
done
NTASKS=$(wc -l < "$QUEUE")
echo "[$(date)] qcmem b-sweep: $NTASKS tasks (${#B_ARR[@]} b × ${#TASK_ARR[@]} task × ${#LEN_ARR[@]} len), j=$RESUME_J lora=${LORA_ADAPTER:-none}"

pop_task() {
  local line=""
  exec 9>"$LOCK"; flock 9
  line="$(head -n 1 "$QUEUE")"
  if [ -n "$line" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
  flock -u 9; exec 9>&-
  echo "$line"
}

run_task_on_group() {
  local gid="$1"; shift
  local gpus=("$@")
  local b="$T_B" task="$T_TASK" L="$T_LEN"
  local _sfx=""; [ "$SELECTOR" != "bm25" ] && _sfx="_${SELECTOR}"
  local run="qcmem_j${RESUME_J}b${b}${_sfx}${RUN_TAG}"
  local results="babilong_results/$run"
  local out_name="${run}_${L}"
  local pids=()
  for si in $(seq 0 $((NSHARD-1))); do
    local g="${gpus[$si]}"
    local shard_tag="_shard${si}of${NSHARD}"
    local csv="$results/$out_name/${task}_${L}${SUFFIX_BASE}${shard_tag}.csv"
    local exprows
    exprows="$($PYBIN -c "print(len(list(range($LIMIT))[$si::$NSHARD])+1)")"
    if [ -f "$csv" ] && [ "$(wc -l < "$csv" 2>/dev/null)" = "$exprows" ]; then
      echo "[skip] $run $task $L shard $si/$NSHARD (complete)"; continue
    fi
    CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_babilong.py \
      --model_path "$MODEL" --resume_j "$RESUME_J" --top_prepay_b "$b" "${LORA_ARG[@]}" \
      --selector "$SELECTOR" --topk "$TOPK" --sink_tokens "$SINK" \
      --results_folder "$results" --output_name "$out_name" \
      --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS" --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" \
      --device cuda:0 --use_instruction --use_examples --use_post_prompt \
      --num_shards "$NSHARD" --shard_index "$si" \
      </dev/null >"$LOGDIR/${run}_${task}_${L}${shard_tag}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

group_worker() {
  local gid="$1"; shift
  local gpus=("$@")
  [ "$gid" -eq 1 ] && sleep 30
  while true; do
    local line; line="$(pop_task)"
    [ -z "$line" ] && break
    read -r T_B T_TASK T_LEN <<< "$line"
    echo "[$(date)] GROUP$gid -> b$T_B $T_TASK $T_LEN"
    run_task_on_group "$gid" "${gpus[@]}"
    echo "[$(date)] GROUP$gid done b$T_B $T_TASK $T_LEN"
  done
  echo "[$(date)] GROUP$gid drained"
}

group_worker 0 "${GROUP0[@]}" &
G0=$!
if [ "$NUM_GROUPS" -ge 2 ]; then
  group_worker 1 "${GROUP1[@]}" &
  G1=$!
  wait "$G0" "$G1"
else
  wait "$G0"
fi

echo "[$(date)] ALL_EVAL_DONE — scoring b-sweep:"
for b in "${B_ARR[@]}"; do
  _sfx=""; [ "$SELECTOR" != "bm25" ] && _sfx="_${SELECTOR}"
  run="qcmem_j${RESUME_J}b${b}${_sfx}${RUN_TAG}"
  echo "=== $run ==="
  $PYBIN scripts/score_nested_babilong.py "babilong_results/$run" --expect -1 2>&1 | tail -6
done
touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] SCHED_DONE"
