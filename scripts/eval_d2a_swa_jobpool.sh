#!/usr/bin/env bash
# D2a job-pool: finish W1+W2 efficiently across GPUs 0-5,7 (GPU6 left for the in-flight W0 32k qa5).
# Skips cells whose CSV already has 100 rows. Greedy: each GPU pulls next job when free.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="http://hy-proxy.woa.com:3128"; export https_proxy="http://hy-proxy.woa.com:3128"; export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_home"
PYBIN="$PROJECT_ROOT/.venv/bin/python"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p11_chunk512_deltarule_normreadout
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter.pt
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPUS=(0 1 2 3 4 5 7)
# Build job list: W in {1,2} x lengths. Put long lengths first (LPT) so they start early.
JOBS=()
for L in 32k 16k 8k 4k 2k 1k 0k; do for W in 1 2; do JOBS+=("$W:$L"); done; done

run_one() {
  local W=$1 L=$2 G=$3
  local RES=babilong_results/p11_chunk512_final_clean_swa${W}
  local LOGD=logs/eval_d2a_clean_swa${W}; mkdir -p "$RES" "$LOGD"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RES --output_name p11_chunk512_final_clean_swa${W}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1
}

# Greedy pool: keep one job per GPU.
declare -A GPU_PID
ji=0
NJOB=${#JOBS[@]}
while (( ji < NJOB )); do
  for G in "${GPUS[@]}"; do
    pid=${GPU_PID[$G]:-}
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      (( ji < NJOB )) || break
      J="${JOBS[$ji]}"; ji=$((ji+1))
      W="${J%%:*}"; L="${J##*:}"
      echo "[$(date)] dispatch W=$W L=$L -> GPU$G"
      run_one "$W" "$L" "$G" &
      GPU_PID[$G]=$!
    fi
  done
  sleep 15
done
wait
echo "[$(date)] JOBPOOL_DONE"
