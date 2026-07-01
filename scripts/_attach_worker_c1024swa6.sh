#!/usr/bin/env bash
# Attach a GROUP1 (GPU 4-7) worker to the ALREADY-RUNNING c1024 swa6 task-pool.
# Pops from the SAME flock queue as the live GROUP0 worker -> no double-run, 2x throughput.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_cache"; export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1
PYBIN="$PROJECT_ROOT/.venv/bin/python"

# mirror the live launch
CK_FILE_ARR=(outputs/T2_recall_chunk1024_N128/mem_space_adapter_step000500.pt outputs/T2_recall_chunk1024_N128/mem_space_adapter.pt)
CK_NAME_ARR=(T2_chunk1024_step500_swa6 T2_chunk1024_final_swa6)
ADAPTER_CONFIG=outputs/T2_recall_chunk1024_N128/adapter_config.json
MODEL=models/Meta-Llama-3-8B
NSHARD=4; LIMIT=100; MAX_NEW_TOKENS=20; CHUNK_SIZE=1024
EXTRA_ARGS="--swa_eval_chunks 6"
SUFFIX_BASE="_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no"
GPUS=(4 5 6 7)

LOGDIR="logs/eval_T2_c1024_swa6_taskpool"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

pop_task() {
  local line=""
  exec 9>"$LOCK"; flock 9
  line="$(head -n 1 "$QUEUE")"
  [ -n "$line" ] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; }
  flock -u 9; exec 9>&-
  echo "$line"
}

while true; do
  line="$(pop_task)"; [ -z "$line" ] && break
  read -r T_CK T_TASK T_LEN <<< "$line"
  run="${CK_NAME_ARR[$T_CK]}"; ckpt="${CK_FILE_ARR[$T_CK]}"
  results="babilong_results/$run"; out_name="${run}_${T_LEN}"
  echo "[$(date)] GROUP1(attach) -> ck$T_CK $T_TASK $T_LEN"
  pids=()
  for si in 0 1 2 3; do
    g="${GPUS[$si]}"; shard_tag="_shard${si}of${NSHARD}"
    csv="$results/$out_name/${T_TASK}_${T_LEN}${SUFFIX_BASE}${shard_tag}.csv"
    exprows="$($PYBIN -c "print(len(list(range($LIMIT))[$si::$NSHARD])+1)")"
    if [ -f "$csv" ] && [ "$(wc -l < "$csv" 2>/dev/null)" = "$exprows" ]; then
      echo "[skip] $run $T_TASK $T_LEN shard $si"; continue
    fi
    CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/run_babilong_mem_space.py \
      --model_path "$MODEL" --checkpoint "$ckpt" --adapter_config "$ADAPTER_CONFIG" \
      --results_folder "$results" --output_name "$out_name" \
      --tasks "$T_TASK" --lengths "$T_LEN" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --batch_size 1 --max_new_tokens "$MAX_NEW_TOKENS" --dtype bfloat16 --attn_impl sdpa \
      --use_instruction --use_examples --use_post_prompt \
      --num_shards "$NSHARD" --shard_index "$si" $EXTRA_ARGS \
      </dev/null >"$LOGDIR/${run}_${T_TASK}_${T_LEN}${shard_tag}.attach.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[$(date)] GROUP1(attach) done ck$T_CK $T_TASK $T_LEN"
done
echo "[$(date)] GROUP1(attach) drained — queue empty"
