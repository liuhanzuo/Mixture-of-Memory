#!/usr/bin/env bash
# 8-GPU sharded likelihood-MC DOWNSTREAM eval -- KNOWLEDGE / COMPREHENSION
# EXTENSION for the OLMo-2 prune-then-heal probe (Paper B, direction #4).
# EVAL-ONLY (forward-only) on .73. Companion to _run_olmo2_probe2_downstream_8gpu.sh
# (which covered the commonsense-reasoning + surface tasks).
#
# New tasks: mmlu (57-subject knowledge, per-subject breakdown), lambada_openai
# (last-word greedy prediction -> long-range coherence), boolq (yes/no reading
# comprehension), commonsense_qa (5-choice), social_iqa (3-choice).
#
# Writes to a SEPARATE output_name suffix "_know" so it does NOT clobber the
# already-DONE 6-task shard files in olmo2_downstream_results/<config>/.
#
# For each config: 1 prepare_data pass (caches datasets, avoids 8-way download
# race) -> fan out 8 shard procs (one per GPU, examples strided [g::8] per task)
# -> wait -> merge shards (acc = sum(correct)/sum(n)).
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python

# outbound proxy for HF datasets download; project data-dir cache (diskB, persists)
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/hf_datasets_cache
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"
BS=8                       # conservative vs default 16: guards 7B long-prompt OOM
DONE=logs/olmo2_downstream_know_DONE
rm -f "$DONE"

# pick the latest 1B keep7 ckpt at launch time (trainer still writing new steps)
LATEST_1B=$(ls -t outputs/olmo2_probe2_1B_keep7fresh2_16card/step*.pt 2>/dev/null | head -1)
LATEST_1B_STEP=$(basename "$LATEST_1B" .pt | sed 's/step//')
echo "[$(date '+%F %T')] latest 1B keep7 ckpt = $LATEST_1B (step $LATEST_1B_STEP)"

# config rows: "output_name|base_model|ckpt(empty=full base)"
CONFIGS=(
  "1B_base_full_know|../models/OLMo-2-0425-1B|"
  "1B_keep7_step50000_know|../models/OLMo-2-0425-1B|outputs/olmo2_probe2_1B_keep7fresh2_16card/step50000.pt"
  "1B_keep7_step100000_know|../models/OLMo-2-0425-1B|outputs/olmo2_probe2_1B_keep7fresh2_16card/step100000.pt"
  "1B_keep7_step147000_know|../models/OLMo-2-0425-1B|outputs/olmo2_probe2_1B_keep7fresh2_16card/step147000.pt"
  "1B_keep7_step${LATEST_1B_STEP}_know|../models/OLMo-2-0425-1B|${LATEST_1B}"
  "7B_base_full_know|../models/OLMo-2-1124-7B|"
  "7B_keep10_step10000_know|../models/OLMo-2-1124-7B|outputs/olmo2_probe2_7B_keep10fresh2/step10000.pt"
)

# prepare datasets once (single process, populates HF cache; resilient per task)
echo "[$(date '+%F %T')] prepare_data (cache all knowledge tasks)"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
  > logs/olmo2_downstream_know_prepare.log 2>&1
tail -12 logs/olmo2_downstream_know_prepare.log

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
