#!/usr/bin/env bash
# Offline BABILong eval for P11 chunk512 (delta_rule writeback + normalize_readout) ablation,
# CONVERGED ckpt (mem_space_adapter.pt = step5000). Same protocol as the step500 eval
# (run_babilong_mem_space.py, qa1/qa2/qa5, 0k-32k, --limit 100, --chunk_size 512, bf16/sdpa);
# ONLY the ckpt (final vs step500) + results/output names differ.
# Runs on disk-B. woa proxy + HF_HOME baked in per the 2026-06-07 07:48 silent-fail lesson
# (diskB has no internet; without proxy BABILong dataset HEAD fails -> 0 samples, no CSV).
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# --- woa proxy + HF cache (diskB no-internet fix) ---
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_home}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p11_chunk512_deltarule_normreadout
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter.pt
RESULTS=babilong_results/p11_chunk512_deltarule_normreadout_final
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_p11_chunk512_deltarule_normreadout_final
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6}"
NG=${#GPUS[@]}

run_on_gpu() {
  local G=$1; shift
  for L in "$@"; do
    echo "[$(date)] GPU $G -> length $L"
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
      --results_folder $RESULTS --output_name p11_chunk512_deltarule_normreadout_final_${L} \
      --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
      --dtype bfloat16 --attn_impl sdpa \
      </dev/null >"$LOGDIR/${L}.log" 2>&1
  done
}

declare -a ASSIGN
for ((i=0; i<${#LENGTHS[@]}; i++)); do
  gi=$((i % NG))
  ASSIGN[$gi]="${ASSIGN[$gi]:-} ${LENGTHS[$i]}"
done
for ((gi=0; gi<NG; gi++)); do
  run_on_gpu "${GPUS[$gi]}" ${ASSIGN[$gi]} &
done
wait
echo "[$(date)] all eval lengths done"
