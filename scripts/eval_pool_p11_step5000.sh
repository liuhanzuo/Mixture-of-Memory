#!/bin/bash
# 8-GPU work-pool driver for P11 chunk512 step5000 BABILong eval.
# Splits (task,length) into 21 jobs across 8 L20A GPUs.
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory

export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy all_proxy

PY=.venv/bin/python
CKPT=outputs/mem_space_p11_chunk512_deltarule_normreadout/mem_space_adapter.pt
CFG=outputs/mem_space_p11_chunk512_deltarule_normreadout/adapter_config.json
MODEL=models/Meta-Llama-3-8B
RUN=p11_chunk512_deltarule_normreadout_step5000
RESROOT=babilong_results/$RUN
LOGDIR=logs/eval_p11_step5000
mkdir -p "$LOGDIR"

TASKS=(qa1 qa2 qa5)
# longest first for better load balancing
LENGTHS=(32k 16k 8k 4k 2k 1k 0k)

# Build job list (task,length)
JOBS=()
for L in "${LENGTHS[@]}"; do
  for T in "${TASKS[@]}"; do
    JOBS+=("$T $L")
  done
done

NGPU=8
i=0
run_job() {
  local gpu=$1 task=$2 length=$3
  local log="$LOGDIR/${task}_${length}.log"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" \
    --checkpoint "$CKPT" \
    --adapter_config "$CFG" \
    --tasks "$task" \
    --lengths "$length" \
    --chunk_size 512 \
    --limit 100 \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --results_folder "$RESROOT" \
    --output_name "${RUN}_${length}" \
    --device cuda:0 \
    > "$log" 2>&1
  echo "DONE gpu=$gpu $task $length rc=$?"
}

# Simple pool: keep <= NGPU jobs in flight, assign round-robin gpu by slot.
declare -A SLOT_PID
free_slots=$(seq 0 $((NGPU-1)))

for job in "${JOBS[@]}"; do
  read -r T L <<< "$job"
  # wait for a free slot
  while :; do
    for s in $(seq 0 $((NGPU-1))); do
      pid=${SLOT_PID[$s]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        run_job "$s" "$T" "$L" &
        SLOT_PID[$s]=$!
        echo "LAUNCH slot=$s gpu=$s $T $L pid=${SLOT_PID[$s]}"
        T=""; break
      fi
    done
    [ -z "$T" ] && break
    sleep 5
  done
done

wait
echo "ALL_EVAL_DONE"
