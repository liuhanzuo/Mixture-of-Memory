#!/usr/bin/env bash
# Offline BABILong eval for F2 long-doc chunk512 FINAL (step5000) ckpt, on diskA (.196).
# Same protocol as eval_p11_..._final.sh. diskA has internet -> proxy harmless/optional.
# Avoids GPU1 (reserved for the eval-speedup parity check running there).
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# woa proxy: BABILong dataset HEAD needs internet; without it -> Network unreachable -> 0 samples, no CSV (silent fail).
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export all_proxy="${all_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="${no_proxy:-mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_home}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/f2_longdoc_chunk512
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter.pt
RESULTS=babilong_results/f2_longdoc_chunk512_final
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_f2_longdoc_chunk512_final
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
read -r -a GPUS <<< "${GPUS:-0 2 3 4 5 6 7}"
NG=${#GPUS[@]}

run_on_gpu() {
  local G=$1; shift
  for L in "$@"; do
    echo "[$(date)] GPU $G -> length $L"
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
      --results_folder $RESULTS --output_name f2_longdoc_chunk512_final_${L} \
      --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
      --dtype bfloat16 --attn_impl sdpa \
      </dev/null >"$LOGDIR/${L}.log" 2>&1
  done
}

declare -a ASSIGN
for ((i=0; i<${#LENGTHS[@]}; i++)); do
  gi=$((i % NG)); ASSIGN[$gi]="${ASSIGN[$gi]:-} ${LENGTHS[$i]}"
done
for ((gi=0; gi<NG; gi++)); do
  run_on_gpu "${GPUS[$gi]}" ${ASSIGN[$gi]} &
done
wait
echo "[$(date)] all eval lengths done"
