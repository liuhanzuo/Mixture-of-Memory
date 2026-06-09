#!/usr/bin/env bash
# P11 chunk512 step500 (SOTA peak) x cross-chunk SWA W0/W1/W2 BABILong eval.
# Local 8x H20 (GPU 0-7), full-speed job-pool. 21 jobs = W{0,1,2} x 7 lengths, qa1/qa2/qa5, n=100.
# LPT ordering: long lengths dispatched first so 8 GPUs stay saturated and slow cells start early.
# Skips cells whose CSV already has >=100 rows (resumable).
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
CKPT=${CKPT_DIR}/mem_space_adapter_step000500.pt
TASKS="qa1 qa2 qa5"
GPUS=(0 1 2 3 4 5 6 7)
# Job list: long lengths first (LPT). For each length, all 3 W values.
JOBS=()
for L in 32k 16k 8k 4k 2k 1k 0k; do for W in 0 1 2; do JOBS+=("$W:$L"); done; done

cell_done() {
  # returns 0 if CSV for (W,L) already has >=100 data rows for all 3 tasks
  local W=$1 L=$2
  local RES=babilong_results/p11_step500_local_swa${W}
  for T in qa1 qa2 qa5; do
    local f
    f=$(ls "$RES"/*"${T}"*"${L}"*.csv 2>/dev/null | head -1)
    [[ -n "$f" ]] || return 1
    local n; n=$(($(wc -l < "$f") - 1))
    (( n >= 100 )) || return 1
  done
  return 0
}

run_one() {
  local W=$1 L=$2 G=$3
  local RES=babilong_results/p11_step500_local_swa${W}
  local LOGD=logs/eval_p11_step500_local_swa${W}; mkdir -p "$RES" "$LOGD"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RES --output_name p11_step500_local_swa${W}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1
}

declare -A GPU_PID
ji=0
NJOB=${#JOBS[@]}
echo "[$(date)] START jobpool: $NJOB jobs over ${#GPUS[@]} GPUs"
while (( ji < NJOB )); do
  for G in "${GPUS[@]}"; do
    pid=${GPU_PID[$G]:-}
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      while (( ji < NJOB )); do
        J="${JOBS[$ji]}"; ji=$((ji+1))
        W="${J%%:*}"; L="${J##*:}"
        if cell_done "$W" "$L"; then
          echo "[$(date)] SKIP (done) W=$W L=$L"
          continue
        fi
        echo "[$(date)] dispatch W=$W L=$L -> GPU$G"
        run_one "$W" "$L" "$G" &
        GPU_PID[$G]=$!
        break
      done
    fi
  done
  (( ji < NJOB )) && sleep 15
done
wait
echo "[$(date)] JOBPOOL_DONE"
