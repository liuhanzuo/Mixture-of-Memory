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

# Build the full (step,length) job list. Order matters for the job-pool scheduler
# below: emit LONGEST lengths first (32k -> 0k) so the slowest jobs start earliest
# and short jobs fill the tail-end gaps, minimizing the long tail.
declare -A LEN_RANK=([0k]=0 [1k]=1 [2k]=2 [4k]=4 [8k]=8 [16k]=16 [32k]=32)
declare -a JOBS=()
for S in $STEPS; do
  for L in "${LENGTHS[@]}"; do
    JOBS+=("${S}:${L}")
  done
done
# Sort JOBS by length descending (longest first). Stable enough: key by LEN_RANK.
if ((${#JOBS[@]} > 1)); then
  mapfile -t JOBS < <(
    for j in "${JOBS[@]}"; do
      L="${j##*:}"; printf '%d\t%s\n' "${LEN_RANK[$L]:-0}" "$j"
    done | sort -rn -k1,1 -s | cut -f2-
  )
fi
echo "[$(date)] sweep: ${#JOBS[@]} jobs ($(echo $STEPS|wc -w) steps x ${#LENGTHS[@]} lengths) over ${#GPUS[@]} GPUs"
echo "[$(date)] job order (longest-first): ${JOBS[*]}"

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

# ---------------------------------------------------------------------------
# Job-pool scheduler (replaces the old batch-barrier `wait`-ALL every nGPU jobs).
#
# Each GPU is a worker slot. We keep all slots busy: launch one job per free GPU,
# then whenever ANY job finishes we immediately hand its GPU to the next pending
# job. This eliminates the old failure mode where a batch of 8 jobs blocked on the
# single slowest one (e.g. 32k) while the other 7 GPUs idled.
#
# bash on the diskB nodes is 4.4.20, so `wait -n` exists but `wait -n -p VAR`
# (which reports WHICH pid finished) needs bash 5.1+. We therefore track a
# pid->gpu association map ourselves and, after each `wait -n` wakeup, reap every
# pid that is no longer alive (`kill -0`) to recover its GPU slot. This is robust
# even if multiple jobs finish between wakeups.
# ---------------------------------------------------------------------------
declare -A PID2GPU=()        # running pid -> gpu it occupies
declare -a FREE_GPUS=("${GPUS[@]}")  # stack of currently-idle GPUs
ji=0                          # index of next pending job
NJOBS=${#JOBS[@]}

reap_finished () {
  # Move any finished pids' GPUs back into FREE_GPUS. Returns # reaped.
  local reaped=0 pid
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null   # collect exit status, avoid zombie
      FREE_GPUS+=("${PID2GPU[$pid]}")
      unset 'PID2GPU[$pid]'
      reaped=$((reaped+1))
    fi
  done
  return 0
}

launch_pending () {
  # Fill every free GPU with the next pending job (skips fire instantly and free
  # the slot again on the next reap). Stops when no free GPU or no pending job.
  while ((${#FREE_GPUS[@]} > 0)) && ((ji < NJOBS)); do
    local job="${JOBS[$ji]}"; ji=$((ji+1))
    local S="${job%%:*}" L="${job##*:}"
    local G="${FREE_GPUS[-1]}"; unset 'FREE_GPUS[-1]'; FREE_GPUS=("${FREE_GPUS[@]}")
    run_one "$S" "$L" "$G" &
    local pid=$!
    PID2GPU[$pid]=$G
    echo "[$(date)] launch job $((ji))/$NJOBS  step=$S len=$L  gpu=$G  pid=$pid  (free_gpus=${#FREE_GPUS[@]})"
  done
}

# Prime all GPU slots, then loop: wait for any job, reap freed slots, refill.
launch_pending
while ((${#PID2GPU[@]} > 0)); do
  wait -n 2>/dev/null    # block until at least one background job exits
  reap_finished
  launch_pending
done
echo "[$(date)] sweep done -> babilong_results/${RESULTS_PREFIX}_step*"
