#!/usr/bin/env bash
# ============================================================================
# 通用 BABILong eval 调度器 — 2-组 task-pool 动态负载均衡 (2026-06-13)
#
# 调度范式(用户指定,codebuddy 标准 eval 方式):
#   * 8 GPU 分 2 组:GROUP0 = GPU 0,1,2,3   GROUP1 = GPU 4,5,6,7
#   * 一个"任务" = (ckpt, task, length),如 step500×qa1×16k,共 100 样本
#   * 一个任务在一组内跑:4 卡各跑 1 个 sample-shard(num_shards=4),即每卡 25 样本
#   * 所有 (ckpt × task × length) 任务进一个共享 pool;哪组空闲就 atomic-pop
#     下一个任务 append 给它 → 最大化吞吐、自动负载均衡(无 LPT 静态预分配)
#   * score_nested_babilong.py 把 4 个 _shard{i}of4 CSV 求和合并回单 cell
#
# 用法(env 传参):
#   RUN_PREFIX=expXXX  CKPT_FILES="ck1.pt ck2.pt"  CK_NAMES="name1 name2" \
#   ADAPTER_CONFIG=outputs/.../adapter_config.json  MODEL=models/Meta-Llama-3-8B \
#   [TASKS="qa1 qa2 qa5"] [LENGTHS="0k 1k 2k 4k 8k 16k 32k"] [NSHARD=4] \
#   [EXTRA_ARGS="--swa_eval_chunks 0"] \
#   bash scripts/_eval_taskpool_2group.sh
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# ---- required params ----
: "${RUN_PREFIX:?set RUN_PREFIX}"
: "${CKPT_FILES:?set CKPT_FILES (space-sep .pt paths)}"
: "${CK_NAMES:?set CK_NAMES (space-sep run names, aligned to CKPT_FILES)}"
: "${ADAPTER_CONFIG:?set ADAPTER_CONFIG}"
MODEL="${MODEL:-models/Meta-Llama-3-8B}"
TASKS="${TASKS:-qa1 qa2 qa5}"
LENGTHS="${LENGTHS:-0k 1k 2k 4k 8k 16k 32k}"
NSHARD="${NSHARD:-4}"                       # shards per task = GPUs per group
CHUNK_SIZE="${CHUNK_SIZE:-512}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SUFFIX_BASE="_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no"

read -r -a CK_FILE_ARR <<< "$CKPT_FILES"
read -r -a CK_NAME_ARR <<< "$CK_NAMES"
read -r -a TASK_ARR <<< "$TASKS"
read -r -a LEN_ARR <<< "$LENGTHS"

# GPU groups (4 GPUs each). Overridable so we can run on a subset of free GPUs.
#   GROUP0_GPUS / GROUP1_GPUS : space-sep GPU ids per group (default 0-3 / 4-7)
#   NUM_GROUPS                : 1 to run a single group (GROUP0_GPUS only), 2 for both
read -r -a GROUP0 <<< "${GROUP0_GPUS:-0 1 2 3}"
read -r -a GROUP1 <<< "${GROUP1_GPUS:-4 5 6 7}"
NUM_GROUPS="${NUM_GROUPS:-2}"

LOGDIR="logs/eval_${RUN_PREFIX}_taskpool"
mkdir -p "$LOGDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

# ---- build task pool: one line per (ckidx, task, length) ----
: > "$QUEUE"
for ckidx in "${!CK_NAME_ARR[@]}"; do
  for task in "${TASK_ARR[@]}"; do
    for L in "${LEN_ARR[@]}"; do
      echo "$ckidx $task $L" >> "$QUEUE"
    done
  done
done
NTASKS=$(wc -l < "$QUEUE")
echo "[$(date)] task-pool built: $NTASKS tasks (${#CK_NAME_ARR[@]} ckpt × ${#TASK_ARR[@]} task × ${#LEN_ARR[@]} len), NSHARD=$NSHARD, NUM_GROUPS=$NUM_GROUPS (G0=${GROUP0[*]} G1=${GROUP1[*]})"

# ---- atomic pop one task line from the queue (flock) ----
pop_task() {
  local line=""
  exec 9>"$LOCK"
  flock 9
  line="$(head -n 1 "$QUEUE")"
  if [ -n "$line" ]; then
    tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  fi
  flock -u 9
  exec 9>&-
  echo "$line"
}

# ---- run one (ckidx,task,L) across a group's 4 GPUs (4 shards in parallel) ----
run_task_on_group() {
  local gid="$1"; shift
  local gpus=("$@")
  local ckidx="$T_CK" task="$T_TASK" L="$T_LEN"
  local run="${CK_NAME_ARR[$ckidx]}"
  local ckpt="${CK_FILE_ARR[$ckidx]}"
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
      echo "[skip] $run $task $L shard $si/$NSHARD (complete)"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/run_babilong_mem_space.py \
      --model_path "$MODEL" --checkpoint "$ckpt" --adapter_config "$ADAPTER_CONFIG" \
      --results_folder "$results" --output_name "$out_name" \
      --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --batch_size 1 --max_new_tokens "$MAX_NEW_TOKENS" \
      --dtype bfloat16 --attn_impl sdpa \
      --use_instruction --use_examples --use_post_prompt \
      --num_shards "$NSHARD" --shard_index "$si" $EXTRA_ARGS \
      </dev/null >"$LOGDIR/${run}_${task}_${L}${shard_tag}.log" 2>&1 &
    pids+=($!)
  done
  # wait for all shards of this task
  for p in "${pids[@]}"; do wait "$p"; done
}

# ---- a group worker: loop popping tasks until queue empty ----
group_worker() {
  local gid="$1"; shift
  local gpus=("$@")
  # stagger group1 start to avoid 8 concurrent 10GB ckpt loads at t=0
  [ "$gid" -eq 1 ] && sleep 30
  while true; do
    local line; line="$(pop_task)"
    [ -z "$line" ] && break
    read -r T_CK T_TASK T_LEN <<< "$line"
    echo "[$(date)] GROUP$gid -> ck$T_CK $T_TASK $T_LEN"
    run_task_on_group "$gid" "${gpus[@]}"
    echo "[$(date)] GROUP$gid done ck$T_CK $T_TASK $T_LEN"
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

echo "[$(date)] ALL_EVAL_DONE — scoring:"
for run in "${CK_NAME_ARR[@]}"; do
  echo "=== $run ==="
  $PYBIN scripts/score_nested_babilong.py "babilong_results/$run" --expect -1 2>&1 | tail -6
done
touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] SCHED_DONE"
