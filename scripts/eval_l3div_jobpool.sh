#!/usr/bin/env bash
# Generic offline BABILong jobpool for L3 diversity sweep ckpts (EXP-1/EXP-3).
# Evals step500 + step1000 (final) of one run, qa1/qa2/qa5, lengths 0k-32k, n=100, chunk512.
# 14 jobs = 2 ckpt x 7 lengths over 8 GPUs (LPT: long lengths first). Resumable.
#
# Env knobs:
#   CKPT_DIR   (required) e.g. outputs/exp1_l3div01
#   RUN        (required) short run name, e.g. exp1_l3div01  -> results=<RUN>_step{500,1000}
#   PROJECT_ROOT (default disk A local)
#   PYTHON_BIN (default .venv)
#   GPUS       (default "0 1 2 3 4 5 6 7")
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="http://hy-proxy.woa.com:3128"; export https_proxy="http://hy-proxy.woa.com:3128"; export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_home"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
: "${CKPT_DIR:?need CKPT_DIR}"; : "${RUN:?need RUN}"
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
TASKS="qa1 qa2 qa5"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
LOGDIR=logs/eval_${RUN}; mkdir -p "$LOGDIR"

# ckpt tag -> file
declare -A CKPT_FILE
CKPT_FILE[step500]=${CKPT_DIR}/mem_space_adapter_step000500.pt
CKPT_FILE[step1000]=${CKPT_DIR}/mem_space_adapter.pt

JOBS=()
for L in 32k 16k 8k 4k 2k 1k 0k; do for S in step500 step1000; do JOBS+=("$S:$L"); done; done

cell_done() {
  local S=$1 L=$2
  local RES=babilong_results/${RUN}_${S}
  for T in qa1 qa2 qa5; do
    local f; f=$(ls "$RES"/*"${T}"*"${L}"*.csv 2>/dev/null | head -1)
    [[ -n "$f" ]] || return 1
    local n; n=$(($(wc -l < "$f") - 1)); (( n >= 100 )) || return 1
  done
  return 0
}

run_one() {
  local S=$1 L=$2 G=$3
  local RES=babilong_results/${RUN}_${S}; mkdir -p "$RES"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint "${CKPT_FILE[$S]}" --adapter_config $ADAPTER_CONFIG \
    --results_folder "$RES" --output_name ${RUN}_${S}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa \
    </dev/null >"$LOGDIR/${S}_${L}.log" 2>&1
}

declare -A GPU_PID
ji=0; NJOB=${#JOBS[@]}
echo "[$(date)] START jobpool ${RUN}: $NJOB jobs over ${#GPUS[@]} GPUs"
while (( ji < NJOB )); do
  for G in "${GPUS[@]}"; do
    pid=${GPU_PID[$G]:-}
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      while (( ji < NJOB )); do
        J="${JOBS[$ji]}"; ji=$((ji+1))
        S="${J%%:*}"; L="${J##*:}"
        if cell_done "$S" "$L"; then echo "[$(date)] SKIP done $S $L"; continue; fi
        echo "[$(date)] dispatch $S $L -> GPU$G"
        run_one "$S" "$L" "$G" &
        GPU_PID[$G]=$!; break
      done
    fi
  done
  (( ji < NJOB )) && sleep 15
done
wait
touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] JOBPOOL_DONE ${RUN}"
