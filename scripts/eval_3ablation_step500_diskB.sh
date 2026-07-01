#!/usr/bin/env bash
# Offline BABILong eval for 3 mem_space step500 ablations on diskB .249.
# Runs P10 / P11 / topk8 each over qa1/qa2/qa5 x 0k-32k (n=100), fanning the
# 21 (run,length) jobs across GPUs 0,1,2,3,7 (avoid GPU4-6: occupied sweep).
# Same protocol as eval_p8b_chunk512_topk8_step500.sh.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL="${MODEL:-models/Meta-Llama-3-8B}"
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
read -r -a GPUS <<< "${GPUS:-0 1 2 3 7}"

# run_name : ckpt_dir : results_prefix
RUNS=(
  "p10:outputs/mem_space_p10_chunk512_keyrep005_stgumbel:p10_keyrep005_stgumbel_step500"
  "p11:outputs/mem_space_p11_chunk512_deltarule_normreadout:p11_deltarule_normreadout_step500"
  "topk8:outputs/mem_space_p8b_chunk512_topk8_diskB:p8b_chunk512_topk8_step500_v2"
)

LOGROOT="logs/eval_3ablation_step500"
mkdir -p "$LOGROOT"

declare -a JOBS=()
for r in "${RUNS[@]}"; do
  for L in "${LENGTHS[@]}"; do
    JOBS+=("${r}|${L}")
  done
done
echo "[$(date)] 3-ablation sweep: ${#JOBS[@]} jobs over ${#GPUS[@]} GPUs (${GPUS[*]})"

run_one () {
  local rname=$1 ckptdir=$2 prefix=$3 len=$4 gpu=$5
  local ckpt="${ckptdir}/mem_space_adapter_step000500.pt"
  local cfg="${ckptdir}/adapter_config.json"
  local results="babilong_results/${prefix}"
  local oname="${prefix}_${len}"
  mkdir -p "$results"
  if [[ ! -f "$ckpt" ]]; then echo "[skip] missing $ckpt"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$ckpt" --adapter_config "$cfg" \
    --results_folder "$results" --output_name "$oname" \
    --tasks $TASKS --lengths "$len" --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa \
    </dev/null >"$LOGROOT/${rname}_${len}.log" 2>&1
}

i=0
for job in "${JOBS[@]}"; do
  r="${job%%|*}"; L="${job##*|}"
  rname="${r%%:*}"; rest="${r#*:}"; ckptdir="${rest%%:*}"; prefix="${rest##*:}"
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  run_one "$rname" "$ckptdir" "$prefix" "$L" "$G" &
  i=$((i+1))
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
echo "[$(date)] 3-ablation sweep done"
