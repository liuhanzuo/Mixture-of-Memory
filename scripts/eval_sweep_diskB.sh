#!/usr/bin/env bash
# General offline BABILong eval sweep for diskB nodes (.76/.249).
# Evaluates one run's MULTIPLE ckpt steps across qa1/qa2/qa5 x 0k-32k, fanning
# (step,length) jobs out over the GPU list so a free 8-GPU node is saturated.
#
# Why this exists: the chunk512 p8_nullsink arm only had step500+step1000 scored
# (the "overtraining" hypothesis check). step1500-4500 ckpts exist but were
# skipped on the assumption they monotonically degrade. This sweep actually
# tests that. Also re-runs evals that previously hung (no proxy/offline + missing
# 16k/32k cache -> now fixed: HF_HUB_OFFLINE + full 0k-32k cache warmed).
#
# Usage:
#   CKPT_DIR=outputs/mem_space_perdoc_chunk512_p8_nullsink_diskB \
#   STEPS="1500 2000 2500 3000 3500 4000 4500" \
#   RESULTS_PREFIX=perdoc_chunk512_p8_nullsink \
#   GPUS="0 1 2 3 4 5 6 7" \
#   bash scripts/eval_sweep_diskB.sh
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# diskB has no internet: read pre-warmed local Arrow cache (0k-32k), never the Hub.
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL="${MODEL:-models/Meta-Llama-3-8B}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR}"
ADAPTER_CONFIG="${CKPT_DIR}/adapter_config.json"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
TASKS="${TASKS:-qa1 qa2 qa5}"
STEPS="${STEPS:?set STEPS e.g. \"1500 2000 2500\"}"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
RESULTS_PREFIX="${RESULTS_PREFIX:?set RESULTS_PREFIX}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"

LOGROOT="logs/eval_sweep_${RESULTS_PREFIX}"
mkdir -p "$LOGROOT"

# Build the full (step,length) job list, then round-robin across GPUs, launching
# one background worker per GPU slot at a time (cap concurrency = #GPUs).
declare -a JOBS=()
for S in $STEPS; do
  for L in "${LENGTHS[@]}"; do
    JOBS+=("${S}:${L}")
  done
done
echo "[$(date)] sweep: ${#JOBS[@]} jobs ($(echo $STEPS|wc -w) steps x ${#LENGTHS[@]} lengths) over ${#GPUS[@]} GPUs"

run_one () {
  local step=$1 len=$2 gpu=$3
  local sname; sname=$(printf '%06d' "$step")
  local ckpt="${CKPT_DIR}/mem_space_adapter_step${sname}.pt"
  local results="babilong_results/${RESULTS_PREFIX}_step${sname}"
  local oname="${RESULTS_PREFIX}_step${sname}_${len}"
  mkdir -p "$results"
  if [[ ! -f "$ckpt" ]]; then echo "[skip] missing $ckpt"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$ckpt" --adapter_config "$ADAPTER_CONFIG" \
    --results_folder "$results" --output_name "$oname" \
    --tasks $TASKS --lengths "$len" --limit 100 --chunk_size "$CHUNK_SIZE" \
    --dtype bfloat16 --attn_impl sdpa \
    </dev/null >"$LOGROOT/step${sname}_${len}.log" 2>&1
}

i=0
for job in "${JOBS[@]}"; do
  S="${job%%:*}"; L="${job##*:}"
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  run_one "$S" "$L" "$G" &
  i=$((i+1))
  # Throttle: once we've filled all GPU slots, wait for the batch to drain.
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
echo "[$(date)] sweep done -> babilong_results/${RESULTS_PREFIX}_step*"
