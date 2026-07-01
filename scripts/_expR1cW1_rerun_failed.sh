#!/usr/bin/env bash
# Re-run the 4 step500 cells that died in the initial concurrent-load SIGKILL burst:
#   qa1 8k (1 item), qa1 16k (2 shards), qa2 16k (2 shards), qa2 32k shard0of4 (1 item)
# Each pinned to its own GPU with a small stagger to avoid re-triggering the load OOM.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
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
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

MODEL="models/Meta-Llama-3-8B"
CKPT_DIR="outputs/expR1cW1_cum_topk32"
ADAPTER_CONFIG="$CKPT_DIR/adapter_config.json"
CKPT="$CKPT_DIR/mem_space_adapter_step000500.pt"
RUN="expR1cW1_cum_topk32_step500"
RESULTS="babilong_results/$RUN"
CHUNK_SIZE=512; LIMIT=100; MAX_NEW_TOKENS=20
LOGDIR="logs/eval_expR1cW1_cum_topk32"
mkdir -p "$LOGDIR"

# (gpu, task, length, shard_index, num_shards)
ITEMS=(
  "0 qa1 8k 0 1"
  "1 qa1 16k 0 2"
  "2 qa1 16k 1 2"
  "3 qa2 16k 0 2"
  "4 qa2 16k 1 2"
  "5 qa2 32k 0 4"
)

run_one() {
  local G=$1 task=$2 L=$3 si=$4 ns=$5
  sleep $(( G * 25 ))
  local out_name="${RUN}_${L}"
  local shardargs=()
  [ "$ns" -gt 1 ] && shardargs=(--num_shards "$ns" --shard_index "$si")
  local shard_tag=""
  [ "$ns" -gt 1 ] && shard_tag="_shard${si}of${ns}"
  echo "[$(date)] GPU $G -> $task $L shard $si/$ns"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ADAPTER_CONFIG" \
    --results_folder "$RESULTS" --output_name "$out_name" \
    --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
    --batch_size 1 --max_new_tokens "$MAX_NEW_TOKENS" \
    --dtype bfloat16 --attn_impl sdpa \
    --use_instruction --use_examples --use_post_prompt \
    "${shardargs[@]}" \
    </dev/null >"$LOGDIR/rerun_${task}_${L}${shard_tag}.log" 2>&1
  echo "[$(date)] GPU $G done $task $L shard $si/$ns"
}

for it in "${ITEMS[@]}"; do
  read -r G task L si ns <<< "$it"
  run_one "$G" "$task" "$L" "$si" "$ns" &
done
wait
echo "[$(date)] RERUN_DONE"
touch "$LOGDIR/RERUN_DONE"
