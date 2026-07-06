#!/usr/bin/env bash
# ============================================================================
# QCMem mid-depth resume — zero-training BABILong j-sweep scheduler
# (2-group task-pool dynamic load balancing, per CLAUDE.md standard eval method)
#
# A "task" = (j, babilong-task, length), e.g. j6 × qa5 × 16k (100 samples).
# 8 GPUs -> 2 groups (GROUP0=0-3, GROUP1=4-7). Each task runs in ONE group
# across its 4 GPUs as 4 sample-shards (num_shards=4, [i::4]). All
# (j × task × length) tasks go into a shared flock'd pool; whichever group is
# idle atomically pops the next -> automatic load balancing.
#
# Output layout matches score_nested_babilong.py:
#   babilong_results/qcmem_j<J>/qcmem_j<J>_<length>/<task>_<length>_<suffix>_shard#of4.csv
#
# Usage (env params):
#   J_VALUES="0 3 6 9 12"  SELECTOR=bm25  TOPK=4 \
#   MODEL=models/Meta-Llama-3-8B \
#   [TASKS="qa1 qa2 qa5"] [LENGTHS="0k 1k 2k 4k 8k 16k"] [NSHARD=4] \
#   [CHUNK_SIZE=512] [LIMIT=100] [SINK=bos] \
#   PROJECT_ROOT=<node root> PYTHON_BIN=<node .venv/conda> \
#   setsid nohup bash scripts/_eval_qcmem_taskpool.sh >logs/qcmem_sweep.out 2>&1 &
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# OFFLINE: no proxy, HF cache local (avoids the .52 offline-load crash).
export http_proxy="" https_proxy="" all_proxy=""
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_cache/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

MODEL="${MODEL:-models/Meta-Llama-3-8B}"
J_VALUES="${J_VALUES:-0 3 6 9 12}"
SELECTOR="${SELECTOR:-bm25}"           # bm25 | recency | oracle
TOPK="${TOPK:-4}"
SINK="${SINK:-bos}"                    # bos | none
TASKS="${TASKS:-qa1 qa2 qa5}"
LENGTHS="${LENGTHS:-0k 1k 2k 4k 8k 16k}"
NSHARD="${NSHARD:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
SUFFIX_BASE="_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no"

read -r -a J_ARR    <<< "$J_VALUES"
read -r -a TASK_ARR <<< "$TASKS"
read -r -a LEN_ARR  <<< "$LENGTHS"
read -r -a GROUP0 <<< "${GROUP0_GPUS:-0 1 2 3}"
read -r -a GROUP1 <<< "${GROUP1_GPUS:-4 5 6 7}"
NUM_GROUPS="${NUM_GROUPS:-2}"

LOGDIR="logs/eval_qcmem_${SELECTOR}_taskpool"
mkdir -p "$LOGDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

# ---- build task pool: one line per (j, task, length) ----
: > "$QUEUE"
for j in "${J_ARR[@]}"; do
  for task in "${TASK_ARR[@]}"; do
    for L in "${LEN_ARR[@]}"; do
      echo "$j $task $L" >> "$QUEUE"
    done
  done
done
NTASKS=$(wc -l < "$QUEUE")
echo "[$(date)] qcmem task-pool: $NTASKS tasks (${#J_ARR[@]} j × ${#TASK_ARR[@]} task × ${#LEN_ARR[@]} len), selector=$SELECTOR topk=$TOPK sink=$SINK NSHARD=$NSHARD NUM_GROUPS=$NUM_GROUPS"

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
  local j="$T_J" task="$T_TASK" L="$T_LEN"
  local _sfx=""; [ "$SELECTOR" != "bm25" ] && _sfx="_${SELECTOR}"
  local run="qcmem_j${j}${_sfx}"
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
      --model_path "$MODEL" --resume_j "$j" \
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
    read -r T_J T_TASK T_LEN <<< "$line"
    echo "[$(date)] GROUP$gid -> j$T_J $T_TASK $T_LEN"
    run_task_on_group "$gid" "${gpus[@]}"
    echo "[$(date)] GROUP$gid done j$T_J $T_TASK $T_LEN"
  done
  echo "[$(date)] GROUP$gid drained — queue empty"
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

echo "[$(date)] ALL_EVAL_DONE — scoring j-sweep:"
for j in "${J_ARR[@]}"; do
  _sfx=""; [ "$SELECTOR" != "bm25" ] && _sfx="_${SELECTOR}"
  run="qcmem_j${j}${_sfx}"
  echo "=== $run ==="
  $PYBIN scripts/score_nested_babilong.py "babilong_results/$run" --expect -1 2>&1 | tail -6
done
touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] SCHED_DONE"
