#!/usr/bin/env bash
# Offline BABILong eval for P11 chunk1024 (delta_rule writeback + normalize_readout), FINAL step5000 ckpt.
# EXACT same protocol as scripts/eval_p11_chunk1024_deltarule_normreadout_step500.sh — ONLY the
# ckpt (final mem_space_adapter.pt) + results/output names + default GPU list + PROJECT_ROOT differ.
# Purpose: confirm whether full 5000-step training fixes the chunk1024 >=1k 断崖 seen at step500
# (step500: qa5 0k-8k=82/43/20/29/16). Runs on disk-A local node (8x H20, all GPUs free).
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p11_chunk1024_deltarule_normreadout
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter.pt
RESULTS=babilong_results/p11_chunk1024_deltarule_normreadout_final
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_p11_chunk1024_deltarule_normreadout_final
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
      --results_folder $RESULTS --output_name p11_chunk1024_deltarule_normreadout_final_${L} \
      --tasks $TASKS --lengths $L --limit 100 --chunk_size 1024 \
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
