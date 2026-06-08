#!/usr/bin/env bash
# Fast offline BABILong eval for mem_space — 8-GPU load-balanced + optional
# cell-internal sample batching (--batch_size). Generalizes
# eval_p11_chunk512_deltarule_normreadout_final.sh:
#   * uses ALL 8 GPUs (the old script used 7 and assigned one length per GPU,
#     so the 32k GPU ran for minutes while the 0k GPU finished in seconds);
#   * flattens every (task, length) cell and greedily packs cells onto GPUs by
#     ESTIMATED cost (cost grows ~linearly with context length, so a 32k cell
#     is weighted far above a 0k cell) → all 8 GPUs finish at about the same
#     time;
#   * BATCH_SIZE>1 enables the batched generation path in
#     run_babilong_mem_space.py (same-chunk-count samples share one forward).
#
# All knobs are env-overridable (sane P11-chunk512 defaults). woa proxy +
# HF_HOME baked in for diskB (no-internet) per the 2026-06-07 silent-fail lesson.
#
# Usage:
#   bash scripts/eval_mem_space_babilong_fast.sh
#   CKPT_DIR=outputs/foo RESULTS=babilong_results/foo CHUNK_SIZE=1024 \
#     BATCH_SIZE=8 LENGTHS="0k 1k 2k 4k 8k 16k 32k" \
#     bash scripts/eval_mem_space_babilong_fast.sh
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# --- woa proxy + HF cache (diskB no-internet fix) ---
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export all_proxy="${all_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="${no_proxy:-mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_home}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# ---- Eval target (override per ckpt) ----
MODEL="${MODEL:-models/Meta-Llama-3-8B}"
CKPT_DIR="${CKPT_DIR:-outputs/mem_space_p11_chunk512_deltarule_normreadout}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:-${CKPT_DIR}/adapter_config.json}"
CKPT="${CKPT:-${CKPT_DIR}/mem_space_adapter.pt}"
RESULTS="${RESULTS:-babilong_results/$(basename "$CKPT_DIR")_fast}"
OUTPREFIX="${OUTPREFIX:-$(basename "$CKPT_DIR")}"
TASKS="${TASKS:-qa1 qa2 qa5}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
LIMIT="${LIMIT:-100}"
# BATCH_SIZE default 1 = the byte-identical correct path. Cell-internal sample
# batching (>1) gives ~1.4x/cell but did NOT pass strict bf16 score parity
# (qa2/2k drifted 27->21 over n=100 — see report), so it is OPT-IN. The 8-GPU
# balanced scheduling below already gives a large wall-clock win on its own.
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
LOGDIR="${LOGDIR:-logs/eval_$(basename "$CKPT_DIR")_fast}"
mkdir -p "$RESULTS" "$LOGDIR"

read -r -a LENGTHS <<< "${LENGTHS:-0k 1k 2k 4k 8k 16k 32k}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NG=${#GPUS[@]}

# ---- Per-length relative cost weight (≈ wall-clock per cell). Generation
# wall-clock is dominated by the number of streamed chunks ≈ length/chunk_size,
# so cost scales ~linearly with the token length. These integer weights are
# used only to balance the greedy bin-packing; exact values don't matter, only
# their ratios. ----
length_weight() {
  case "$1" in
    0k)  echo 1   ;;
    1k)  echo 2   ;;
    2k)  echo 4   ;;
    4k)  echo 8   ;;
    8k)  echo 16  ;;
    16k) echo 32  ;;
    32k) echo 64  ;;
    *)   echo 8   ;;  # unknown → mid weight
  esac
}

# ---- Build the flat list of (task,length) cells with weights, sort desc by
# weight (longest-processing-time-first), greedily assign each to the currently
# least-loaded GPU. This is the classic LPT makespan heuristic. ----
declare -a CELL_TASK CELL_LEN CELL_W
ci=0
for task in $TASKS; do
  for L in "${LENGTHS[@]}"; do
    CELL_TASK[$ci]="$task"
    CELL_LEN[$ci]="$L"
    CELL_W[$ci]="$(length_weight "$L")"
    ci=$((ci+1))
  done
done
NCELLS=$ci

# Sort cell indices by weight descending (simple selection on small arrays).
declare -a ORDER
for ((i=0;i<NCELLS;i++)); do ORDER[$i]=$i; done
for ((a=0;a<NCELLS;a++)); do
  best=$a
  for ((b=a+1;b<NCELLS;b++)); do
    if (( CELL_W[ORDER[b]] > CELL_W[ORDER[best]] )); then best=$b; fi
  done
  tmp=${ORDER[$a]}; ORDER[$a]=${ORDER[$best]}; ORDER[$best]=$tmp
done

# Greedy LPT assignment.
declare -a GPU_LOAD GPU_CELLS
for ((g=0;g<NG;g++)); do GPU_LOAD[$g]=0; GPU_CELLS[$g]=""; done
for ((o=0;o<NCELLS;o++)); do
  c=${ORDER[$o]}
  # find least-loaded gpu
  ming=0
  for ((g=1;g<NG;g++)); do
    if (( GPU_LOAD[g] < GPU_LOAD[ming] )); then ming=$g; fi
  done
  GPU_LOAD[$ming]=$(( GPU_LOAD[ming] + CELL_W[c] ))
  GPU_CELLS[$ming]="${GPU_CELLS[$ming]} $c"
done

echo "[$(date)] eval_mem_space_babilong_fast: ckpt=$CKPT chunk=$CHUNK_SIZE bsz=$BATCH_SIZE"
echo "[$(date)] cells=$NCELLS over ${NG} GPUs; per-GPU weighted load:"
for ((g=0;g<NG;g++)); do
  echo "  GPU ${GPUS[$g]} load=${GPU_LOAD[$g]} cells:${GPU_CELLS[$g]}"
done

run_gpu_cells() {
  local G=$1; shift
  local cells=("$@")
  for c in "${cells[@]}"; do
    local task="${CELL_TASK[$c]}"
    local L="${CELL_LEN[$c]}"
    echo "[$(date)] GPU $G -> task $task length $L (w=${CELL_W[$c]})"
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ADAPTER_CONFIG" \
      --results_folder "$RESULTS" --output_name "${OUTPREFIX}_${task}_${L}" \
      --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --batch_size "$BATCH_SIZE" --max_new_tokens "$MAX_NEW_TOKENS" \
      --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" \
      </dev/null >"$LOGDIR/${task}_${L}.log" 2>&1
  done
}

for ((g=0;g<NG;g++)); do
  # shellcheck disable=SC2086
  read -r -a cells_arr <<< "${GPU_CELLS[$g]}"
  if [ ${#cells_arr[@]} -gt 0 ]; then
    run_gpu_cells "${GPUS[$g]}" "${cells_arr[@]}" &
  fi
done
wait
echo "[$(date)] all eval cells done -> $RESULTS"
