#!/usr/bin/env bash
# Generic mem_space BABILong sharded LPT scheduler over 8 GPUs.
# Parametrized by env vars so one script serves multiple ckpts/runs:
#   CKPT_FILE      (required) path to adapter .pt
#   ADAPTER_CONFIG (required) path to adapter_config.json
#   RESULTS        (required) babilong_results/<run>
#   OUT_PREFIX     (required) output_name prefix, e.g. expW2_softwrite002_step500
#   LOGDIR         (required) logs/eval_<run>
#   CHUNK_SIZE     (default 512)
# 1 ckpt x 3 tasks x 7 lengths = 21 cells. Heavy lengths sample-sharded
# (32k->4, 16k->2 shards/task) => 33 work items LPT-balanced over 8 GPUs.
# Shards of same (task,len) share one output_name dir (_shard{i}of{N} suffix);
# score_nested_babilong.py globs+sums them.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYBIN="$PROJECT_ROOT/.venv/bin/python"

MODEL="models/Meta-Llama-3-8B"
: "${CKPT_FILE:?need CKPT_FILE}"
: "${ADAPTER_CONFIG:?need ADAPTER_CONFIG}"
: "${RESULTS:?need RESULTS}"
: "${OUT_PREFIX:?need OUT_PREFIX}"
: "${LOGDIR:?need LOGDIR}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
LIMIT=100
MAX_NEW_TOKENS=20
SUFFIX_BASE="_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no"
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
TASKS=(qa1 qa2 qa5)

nshards_for() { case "$1" in 32k) echo 4;; 16k) echo 2;; *) echo 1;; esac; }
chunks_for()  { case "$1" in 0k) echo 1;; 1k) echo 2;; 2k) echo 4;; 4k) echo 8;; 8k) echo 16;; 16k) echo 32;; 32k) echo 64;; *) echo 8;; esac; }

declare -a I_TASK I_LEN I_SI I_NS I_W
ni=0
for task in "${TASKS[@]}"; do
  for L in "${LENGTHS[@]}"; do
    NS="$(nshards_for "$L")"
    W=$(( $(chunks_for "$L") / NS )); (( W < 1 )) && W=1
    for ((si=0; si<NS; si++)); do
      I_TASK[$ni]="$task"; I_LEN[$ni]="$L"; I_SI[$ni]="$si"; I_NS[$ni]="$NS"; I_W[$ni]="$W"
      ni=$((ni+1))
    done
  done
done
NItems=$ni
echo "[$(date)] $NItems work items"

declare -a ORDER
for ((i=0;i<NItems;i++)); do ORDER[$i]=$i; done
for ((a=0;a<NItems;a++)); do
  best=$a
  for ((b=a+1;b<NItems;b++)); do
    if (( I_W[ORDER[b]] > I_W[ORDER[best]] )); then best=$b; fi
  done
  t=${ORDER[$a]}; ORDER[$a]=${ORDER[$best]}; ORDER[$best]=$t
done

NG=8
declare -a GPU_LOAD GPU_ITEMS
for ((g=0;g<NG;g++)); do GPU_LOAD[$g]=0; GPU_ITEMS[$g]=""; done
for ((o=0;o<NItems;o++)); do
  c=${ORDER[$o]}
  ming=0
  for ((g=1;g<NG;g++)); do
    if (( GPU_LOAD[g] < GPU_LOAD[ming] )); then ming=$g; fi
  done
  GPU_LOAD[$ming]=$(( GPU_LOAD[ming] + I_W[c] ))
  GPU_ITEMS[$ming]="${GPU_ITEMS[$ming]} $c"
done

echo "[$(date)] $NItems items over $NG GPUs (run=$OUT_PREFIX)"
for ((g=0;g<NG;g++)); do echo "  GPU $g load=${GPU_LOAD[$g]} items:${GPU_ITEMS[$g]}"; done

run_gpu_items() {
  local G=$1; shift
  local items=("$@")
  for c in "${items[@]}"; do
    local task="${I_TASK[$c]}"; local L="${I_LEN[$c]}"
    local si="${I_SI[$c]}"; local ns="${I_NS[$c]}"
    local out_name="${OUT_PREFIX}_${L}"
    local shard_tag=""
    [ "$ns" -gt 1 ] && shard_tag="_shard${si}of${ns}"
    local csv="$RESULTS/$out_name/${task}_${L}${SUFFIX_BASE}${shard_tag}.csv"
    local exprows
    exprows="$($PYBIN -c "print(len(list(range($LIMIT))[$si::$ns])+1)")"
    if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -eq "$exprows" ]; then
      echo "[skip-complete] $task $L shard $si/$ns"
      continue
    fi
    echo "[$(date)] GPU $G -> $task $L shard $si/$ns (w=${I_W[$c]})"
    local shardargs=()
    [ "$ns" -gt 1 ] && shardargs=(--num_shards "$ns" --shard_index "$si")
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path "$MODEL" --checkpoint "$CKPT_FILE" --adapter_config "$ADAPTER_CONFIG" \
      --results_folder "$RESULTS" --output_name "$out_name" \
      --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --batch_size 1 --max_new_tokens "$MAX_NEW_TOKENS" \
      --dtype bfloat16 --attn_impl sdpa \
      --use_instruction --use_examples --use_post_prompt \
      "${shardargs[@]}" \
      </dev/null >"$LOGDIR/${task}_${L}${shard_tag}.log" 2>&1
  done
}

for ((g=0;g<NG;g++)); do
  read -r -a items_arr <<< "${GPU_ITEMS[$g]}"
  if [ ${#items_arr[@]} -gt 0 ]; then
    run_gpu_items "$g" "${items_arr[@]}" &
  fi
done
wait
echo "[$(date)] ALL_EVAL_DONE -> $RESULTS"
touch "$LOGDIR/SCHED_DONE"
