#!/usr/bin/env bash
# HNST position-stratified decisive eval: 3 arms x lengths, 4-GPU job pool.
#   arms: tree (HNST), b25 (plain FIFO eviction@25), oracle (upper bound)
# Each (arm,length) cell is 4-way sharded; a pool keeps <=NGPU jobs live,
# one per free GPU, pulling tasks from a queue. All arms record needle_pos.
set -uo pipefail
R=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT="$R/outputs/mem_space_fifo_b50_chunk512/full_model.pt"
ACFG="$R/outputs/mem_space_fifo_b50_chunk512/adapter_config.json"
MODEL="$R/models/Meta-Llama-3-8B"
OUT="$R/babilong_results/HNST_posstrat"
LOGD="$R/logs/hnst"; mkdir -p "$LOGD"

GPUS=(${GPUS:-4 5 6 7})           # free GPUs
LENGTHS=(${LENGTHS:-16k 32k})
LIMIT=${LIMIT:-100}
NSHARD=${NSHARD:-4}

COMMON="--tasks qa5 --chunk_size 512 --batch_size 1 --max_new_tokens 20 \
  --dtype bfloat16 --attn_impl sdpa --use_instruction --use_examples \
  --use_post_prompt --swa_eval_chunks 0 --record_needle_pos --limit $LIMIT"

arm_extra() {
  case "$1" in
    tree)   echo "--fifo_keep_set_mode tree --fifo_tree_branch 8 --fifo_tree_beam 3 --fifo_keep_topk 25 --fifo_keep_recency 2 --fifo_keep_all_buffer" ;;
    b25)    echo "--fifo_buffer_chunks_eval 25" ;;   # amnesia baseline: evict to 25, attend all buffered
    oracle) echo "--fifo_keep_set_mode oracle --fifo_keep_recency 2 --fifo_keep_all_buffer" ;;
  esac
}

# Build task queue: "arm length shard"
QUEUE=()
for L in "${LENGTHS[@]}"; do
  for A in tree b25 oracle; do
    for ((s=0; s<NSHARD; s++)); do QUEUE+=("$A $L $s"); done
  done
done

run_task() {
  local gpu=$1 arm=$2 L=$3 si=$4
  local extra; extra=$(arm_extra "$arm")
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder "$OUT" --output_name "${arm}/${arm}_${L}" \
    --lengths "$L" --num_shards "$NSHARD" --shard_index "$si" \
    $COMMON $extra > "$LOGD/${arm}_${L}_s${si}.log" 2>&1
  echo "DONE $arm $L s$si (gpu$gpu)"
}

# Job pool: one slot per GPU.
declare -A SLOT_PID
qi=0
NG=${#GPUS[@]}
while (( qi < ${#QUEUE[@]} )) || (( ${#SLOT_PID[@]} > 0 )); do
  # fill free slots
  for g in "${GPUS[@]}"; do
    if (( qi >= ${#QUEUE[@]} )); then break; fi
    if [[ -z "${SLOT_PID[$g]:-}" ]]; then
      read -r A L S <<< "${QUEUE[$qi]}"
      run_task "$g" "$A" "$L" "$S" &
      SLOT_PID[$g]=$!
      echo "[pool] launched $A $L s$S on gpu$g pid=${SLOT_PID[$g]} (task $((qi+1))/${#QUEUE[@]})"
      qi=$((qi+1))
    fi
  done
  # reap finished
  sleep 15
  for g in "${GPUS[@]}"; do
    pid="${SLOT_PID[$g]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      unset SLOT_PID[$g]
    fi
  done
done
echo "HNST_POSSTRAT_ALL_DONE"
