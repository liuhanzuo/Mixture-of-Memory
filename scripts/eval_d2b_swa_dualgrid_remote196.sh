#!/usr/bin/env bash
# EVAL-5 / D2b dual-protocol BABILong eval on diskA REMOTE .196 (8x H20).
# Evaluates the D2b train-side-SWA run (outputs/d2b_swa_train_w2) under TWO eval
# protocols, on TWO checkpoints:
#   ckpt  in {step500, final(=step5000)}
#   W     in {0 (standard single-chunk), 2 (--swa_eval_chunks 2)}
# => 4 grids x 7 lengths = 28 cells. All cells bsz=1 (W2 only supports bs1; keep
# W0 bs1 too for byte-exact final numbers).
# Greedy GPU pool over GPUs 0-7: each GPU pulls the next job when free.
# Skips cells whose CSV already has >=100 rows (resumable).
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# diskA has .hf_home cache with all 7 lengths -> try offline first; proxy as fallback.
export HF_HOME="$PROJECT_ROOT/.hf_home"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/d2b_swa_train_w2
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPUS=(0 1 2 3 4 5 6 7)

# checkpoint label -> file
declare -A CKPT_FILE
CKPT_FILE[step500]=${CKPT_DIR}/mem_space_adapter_step000500.pt
CKPT_FILE[step5000]=${CKPT_DIR}/mem_space_adapter.pt

# Build job list: (ckpt, W, length). LPT: long lengths first so they start early.
JOBS=()
for L in 32k 16k 8k 4k 2k 1k 0k; do
  for CK in step500 step5000; do
    for W in 0 2; do
      JOBS+=("$CK:$W:$L")
    done
  done
done

cell_done() {
  # returns 0 if the CSV for this grid+length already has >=100 data rows for all 3 tasks
  local NAME=$1 L=$2
  local RES=babilong_results/$NAME
  local sub=$RES/${NAME}_${L}
  [[ -d "$sub" ]] || return 1
  local t
  for t in qa1 qa2 qa5; do
    local f
    f=$(ls "$sub"/${t}_${L}_*.csv 2>/dev/null | head -1)
    [[ -n "$f" ]] || return 1
    local n
    n=$(( $(wc -l < "$f") - 1 ))
    (( n >= 100 )) || return 1
  done
  return 0
}

run_one() {
  local CK=$1 W=$2 L=$3 G=$4
  local NAME=d2b_${CK}_W${W}
  local RES=babilong_results/$NAME
  local LOGD=logs/eval_d2b_${CK}_W${W}; mkdir -p "$RES" "$LOGD"
  if cell_done "$NAME" "$L"; then
    echo "[$(date)] SKIP (done) $NAME $L -> GPU$G"
    return 0
  fi
  echo "[$(date)] dispatch $CK W=$W L=$L -> GPU$G  (out=$NAME)"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint "${CKPT_FILE[$CK]}" --adapter_config $ADAPTER_CONFIG \
    --results_folder $RES --output_name ${NAME}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 --batch_size 1 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1
}

declare -A GPU_PID
ji=0
NJOB=${#JOBS[@]}
echo "[$(date)] D2B_DUALGRID_START njobs=$NJOB gpus=${GPUS[*]}"
while (( ji < NJOB )); do
  for G in "${GPUS[@]}"; do
    pid=${GPU_PID[$G]:-}
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      (( ji < NJOB )) || break
      J="${JOBS[$ji]}"; ji=$((ji+1))
      CK="${J%%:*}"; rest="${J#*:}"; W="${rest%%:*}"; L="${rest##*:}"
      run_one "$CK" "$W" "$L" "$G" &
      GPU_PID[$G]=$!
    fi
  done
  sleep 15
done
wait
echo "[$(date)] D2B_DUALGRID_DONE"
