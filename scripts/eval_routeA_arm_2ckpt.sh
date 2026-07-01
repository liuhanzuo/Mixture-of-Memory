#!/usr/bin/env bash
# Generic offline BABILong eval driver for one ROUTE-A arm, BOTH step500 + step1000.
# Distributes (ckpt x length) = 2 x 7 = 14 units across 8 GPUs (each GPU runs its
# assigned units sequentially; one 8B model per GPU at a time). Same protocol as
# eval_p11_chunk512_deltarule_normreadout_step500.sh: qa1/qa2/qa5, lengths 0k-32k,
# --limit 100, --chunk_size 512, bfloat16/sdpa. Arch flags from adapter_config.json.
#
# Required env:
#   ARM        e.g. routeA_arm1   (used for output_name / results / scores)
#   CKPT_DIR   e.g. outputs/routeA_arm1_lossfree001
# Optional env:
#   PROJECT_ROOT (default = diskA local root)
#   PYTHON_BIN   (default = $PROJECT_ROOT/.venv/bin/python)
#   GPUS         (default "0 1 2 3 4 5 6 7")
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
: "${ARM:?must set ARM}"
: "${CKPT_DIR:?must set CKPT_DIR}"
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NG=${#GPUS[@]}

# step -> ckpt file
declare -A CKPT
CKPT[step500]=${CKPT_DIR}/mem_space_adapter_step000500.pt
CKPT[step1000]=${CKPT_DIR}/mem_space_adapter.pt

run_unit() {
  local G=$1 STEP=$2 L=$3
  local OUT=${ARM}_${STEP}
  local RESULTS=babilong_results/${OUT}
  local LOGDIR=logs/eval_${OUT}
  mkdir -p "$RESULTS" "$LOGDIR"
  echo "[$(date)] GPU $G -> $STEP $L"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint "${CKPT[$STEP]}" --adapter_config $ADAPTER_CONFIG \
    --results_folder "$RESULTS" --output_name "${OUT}_${L}" \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa \
    </dev/null >"$LOGDIR/${STEP}_${L}.log" 2>&1
}

# Build the 14-unit job list: step500 lengths first, then step1000.
UNITS=()
for STEP in step500 step1000; do
  for L in "${LENGTHS[@]}"; do UNITS+=("${STEP}:${L}"); done
done

# Round-robin assign units to GPUs.
declare -a ASSIGN
for ((i=0; i<${#UNITS[@]}; i++)); do
  gi=$((i % NG))
  ASSIGN[$gi]="${ASSIGN[$gi]:-} ${UNITS[$i]}"
done

worker() {
  local G=$1; shift
  for U in "$@"; do
    STEP=${U%%:*}; L=${U##*:}
    run_unit "$G" "$STEP" "$L"
  done
}

for ((gi=0; gi<NG; gi++)); do
  worker "${GPUS[$gi]}" ${ASSIGN[$gi]} &
done
wait
echo "[$(date)] ALL eval units done for $ARM"
