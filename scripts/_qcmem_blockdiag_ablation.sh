#!/usr/bin/env bash
# QCMem ablation: (i) standard full-attention read vs (ii) block-diagonal read
# (reuse per-chunk query-blind KV). Same j=12, topk12, bm25, Qwen3-8B + distill
# LoRA. Tasks: niah_single + niah_multikey, lengths 8k + 16k. n=50 per cell.
# Each cell sharded into NUM_SHARDS pieces; 8 GPUs pull (arm,task,len,shard)
# units from one shared flock'd pool -> dynamic load balance, no idle GPU.
set -u

RD=${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}
PYBIN=${PYTHON_BIN:-$RD/.venv/bin/python}
cd "$RD" || exit 1
export WANDB_MODE=offline
export HF_HOME=$RD/.hf_cache HF_DATASETS_CACHE=$RD/.hf_cache/datasets

MODEL=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
LORA=outputs/qcmem_distill_qwen_j12_r32_4k/final
RESULTS=ruler_results/qcmem_blockdiag_ablation
NS=${NUM_SAMPLES:-50}
NSHARD=${NUM_SHARDS:-4}
TASKS=(niah_single niah_multi)
LENGTHS=(8k 16k)
ARMS=(standard blockdiag)

mkdir -p logs "$RESULTS"
POOL=$(mktemp); LOCK=$(mktemp)
# 16k is the slow tail -> enqueue it first so it starts earliest.
for a in "${ARMS[@]}"; do for t in "${TASKS[@]}"; do
  for L in 16k 8k; do
    for s in $(seq 0 $((NSHARD-1))); do echo "$a $t $L $s"; done
  done
done; done > "$POOL"

pop() {
  local line
  exec 9>"$LOCK"; flock 9
  line=$(head -n1 "$POOL")
  [ -n "$line" ] && sed -i '1d' "$POOL"
  flock -u 9
  printf '%s' "$line"
}

worker() {
  local gpu=$1 cell arm t L s flag oname
  while true; do
    cell=$(pop); [ -z "$cell" ] && break
    arm=$(echo "$cell" | awk '{print $1}')
    t=$(echo "$cell"   | awk '{print $2}')
    L=$(echo "$cell"   | awk '{print $3}')
    s=$(echo "$cell"   | awk '{print $4}')
    flag=""; [ "$arm" = "blockdiag" ] && flag="--reuse_kv_blockdiag"
    oname="qcmem_${arm}_j12"
    echo "[$(date +%H:%M:%S)] gpu$gpu -> arm=$arm task=$t len=$L shard=$s/$NSHARD" >> logs/blockdiag_ablation_sched.out
    CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_ruler_qcmem.py \
      --model_path "$MODEL" --resume_j 12 --lora_adapter "$LORA" \
      --selector bm25 --topk 12 --sink_tokens bos $flag \
      --ruler_tasks "$t" --lengths "$L" --limit "$NS" \
      --num_shards "$NSHARD" --shard_index "$s" \
      --output_name "$oname" --results_folder "$RESULTS" \
      >>"logs/blockdiag_${arm}_gpu${gpu}.log" 2>&1
  done
}

echo "[$(date)] launching blockdiag ablation, pool=$(wc -l <"$POOL") units, 8 GPUs, nshard=$NSHARD" \
  | tee -a logs/blockdiag_ablation_sched.out
for g in 0 1 2 3 4 5 6 7; do worker "$g" & done
wait
echo "[$(date)] ALL EVAL DONE -> aggregating" | tee -a logs/blockdiag_ablation_sched.out
$PYBIN scripts/aggregate_blockdiag_ablation.py --results_folder "$RESULTS" \
  | tee -a logs/blockdiag_ablation_sched.out
echo "[$(date)] ALL DONE" | tee -a logs/blockdiag_ablation_sched.out
