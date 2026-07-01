#!/usr/bin/env bash
# Offline BABILong eval scheduler for EXP-R1 (dead-slot recycling, R1 = P11 chunk512
# base + dead_slot_reset_interval 8). Two checkpoints (step500 + step1000), qa1/qa2/qa5,
# lengths 0k-32k, --limit 100, --chunk_size 512, bfloat16/sdpa.
#
# Runs on .196 (disk-A, share_303098609) with conda torch-base python.
#
# Parallelism: a flock-protected atomic job queue feeds 8 GPU workers. Heavy lengths
# (8k/16k/32k) are split into sample-level stride shards (--num_shards/--shard_index)
# so all 8 GPUs stay saturated; score_nested_babilong.py globs+sums the shard CSVs
# back into a single cell score.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/expR1_deadslot_r8
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
TASKS="qa1 qa2 qa5"

# Two checkpoints
CK_NAME=(expR1_deadslot_step500 expR1_deadslot_step1000)
CK_FILE=("${CKPT_DIR}/mem_space_adapter_step000500.pt" "${CKPT_DIR}/mem_space_adapter.pt")

LOGDIR=logs/eval_expR1_deadslot
mkdir -p "$LOGDIR"

# Per-length shard counts (heavy lengths split for GPU saturation)
declare -A NSHARD=( [0k]=1 [1k]=1 [2k]=1 [4k]=1 [8k]=2 [16k]=4 [32k]=4 )
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)

# Build job list: "ckidx|len|nshards|shardidx", heaviest first (LPT-ish ordering)
JOBFILE="$LOGDIR/jobs.txt"
: > "$JOBFILE"
# emit in heavy->light length order so the queue front is the expensive work
for L in 32k 16k 8k 4k 2k 1k 0k; do
  N=${NSHARD[$L]}
  for CK in 0 1; do
    for ((s=0; s<N; s++)); do
      echo "${CK}|${L}|${N}|${s}" >> "$JOBFILE"
    done
  done
done
NJOBS=$(wc -l < "$JOBFILE")
echo "[$(date)] total jobs: $NJOBS"

CNTFILE="$LOGDIR/.counter"
LCKFILE="$LOGDIR/.counter.lock"
echo 0 > "$CNTFILE"

next_idx() {
  local idx
  exec 9>"$LCKFILE"
  flock 9
  idx=$(cat "$CNTFILE")
  echo $((idx + 1)) > "$CNTFILE"
  flock -u 9
  exec 9>&-
  echo "$idx"
}

run_one() {
  local G=$1 spec=$2
  IFS='|' read -r CK L N S <<< "$spec"
  local run="${CK_NAME[$CK]}"
  local ckpt="${CK_FILE[$CK]}"
  local results="babilong_results/${run}"
  local oname="${run}_${L}"
  local shardflag=""
  local tag=""
  if [ "$N" -gt 1 ]; then
    shardflag="--num_shards $N --shard_index $S"
    tag="_shard${S}of${N}"
  fi
  echo "[$(date)] GPU $G -> ck=$run len=$L shard=$S/$N"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint "$ckpt" --adapter_config $ADAPTER_CONFIG \
    --results_folder "$results" --output_name "$oname" \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa $shardflag \
    </dev/null >"$LOGDIR/${run}_${L}${tag}.log" 2>&1
}

worker() {
  local G=$1
  while true; do
    local idx; idx=$(next_idx)
    if [ "$idx" -ge "$NJOBS" ]; then break; fi
    local spec; spec=$(sed -n "$((idx + 1))p" "$JOBFILE")
    run_one "$G" "$spec"
  done
  echo "[$(date)] worker GPU $G done"
}

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
for G in "${GPUS[@]}"; do
  worker "$G" &
done
wait
echo "[$(date)] ALL EVAL DONE"
