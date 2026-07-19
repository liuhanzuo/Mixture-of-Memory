#!/usr/bin/env bash
# 8-GPU sharded likelihood-MC DOWNSTREAM eval -- KNOWLEDGE / COMPREHENSION,
# single config: the APEX prune-heal point OLMo-2-7B keep14+fresh2
# (16L/32=50%), step128000 (fully converged heal). EVAL-ONLY on .73.
# Same harness as scripts/_run_olmo2_probe2_downstream_know_8gpu.sh (verbatim
# 8-shard [g::8] + merge; _know output_name so it never clobbers core shards).
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/hf_datasets_cache
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"
BS=8                       # conservative vs default 16: guards 7B long-prompt OOM
DONE=logs/olmo2_downstream_keep14_know_DONE
rm -f "$DONE"

# config rows: "output_name|base_model|ckpt(empty=full base)"
CONFIGS=(
  "7B_keep14_step128000_know|../models/OLMo-2-1124-7B|outputs/olmo2_probe2_7B_keep14fresh2/step128000.pt"
)

echo "[$(date '+%F %T')] prepare_data (cache all knowledge tasks)"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
  > logs/olmo2_downstream_keep14_know_prepare.log 2>&1
tail -12 logs/olmo2_downstream_keep14_know_prepare.log

for row in "${CONFIGS[@]}"; do
  NAME="${row%%|*}"; rest="${row#*|}"
  BASE="${rest%%|*}"; CKPT="${rest#*|}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] CONFIG $NAME base=$BASE ckpt='${CKPT:-<FULL BASE>}'"
  CKARG=""; [ -n "$CKPT" ] && CKARG="--ckpt $CKPT"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" $CKARG --tasks "$TASKS" \
      --num_shards 8 --shard_index $g --batch_size $BS \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] $NAME shards done; merging"
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
done

echo "[$(date '+%F %T')] ALL CONFIGS DONE" | tee "$DONE"
for row in "${CONFIGS[@]}"; do
  NAME="${row%%|*}"
  echo "--- $NAME ---" >> "$DONE"
  cat "olmo2_downstream_results/${NAME}/summary.json" >> "$DONE" 2>/dev/null
done
echo "[$(date '+%F %T')] wrote $DONE"
