#!/usr/bin/env bash
# Offline BABILong eval for P11 chunk512 + L3-summary recon-token-weight=1.0 ablation, step500.
# SAME protocol as eval_p11_chunk512_deltarule_normreadout (the P11 baseline) but ckpt comes
# from the l3_recon_token_weight sweep arm w=1.0. chunk_size=512. ckpt lives on disk B already.
# qa1/qa2/qa5, lengths 0k-32k, --limit 100, bfloat16/sdpa. arch flags from adapter_config.json.
#
# Lengths distributed across GPUS; each GPU runs its lengths sequentially. Default GPUS="1 2".
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p11_chunk512_l3recontoken_w0.3
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step000500.pt
RESULTS=babilong_results/p11_chunk512_l3recontoken_w0.3_step500
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_p11_chunk512_l3recontoken_w0.3_step500
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
read -r -a GPUS <<< "${GPUS:-1 2}"
NG=${#GPUS[@]}

run_on_gpu() {
  local G=$1; shift
  for L in "$@"; do
    echo "[$(date)] GPU $G -> length $L"
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
      --results_folder $RESULTS --output_name p11_chunk512_l3recontoken_w0.3_step500_${L} \
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
