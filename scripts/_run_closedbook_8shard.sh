#!/usr/bin/env bash
# Paper B P0.3 closed-book QA: run ONE model as 8 GPU shards then merge.
# Usage:
#   OUTPUT_NAME=base_full MODEL_ARGS="" bash scripts/_run_closedbook_8shard.sh
#   OUTPUT_NAME=keep14_step200k \
#     MODEL_ARGS="--ckpt outputs/.../step200000.pt --keep_front_layers 14 --n_fresh_layers 2" \
#     bash scripts/_run_closedbook_8shard.sh
# Env overrides: PYTHON_BIN, BASE_MODEL, TASKS, BATCH_SIZE, RESULTS_ROOT
set -u
W=${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}
cd "$W" || exit 3
PY=${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}
BASE=${BASE_MODEL:-../models/OLMo-2-1124-7B}
OUT=${OUTPUT_NAME:?need OUTPUT_NAME}
EXTRA="${MODEL_ARGS:-}"
TASKS=${TASKS:-popqa,triviaqa}
BS=${BATCH_SIZE:-32}
RROOT=${RESULTS_ROOT:-olmo2_closedbook_results}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p logs "$RROOT/$OUT"
echo "[$(date '+%F %T')] START $OUT  base=$BASE  extra=[$EXTRA]  bs=$BS  tasks=$TASKS"
pids=()
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
    --base_model "$BASE" $EXTRA \
    --tasks "$TASKS" --num_shards 8 --shard_index "$g" \
    --batch_size "$BS" --add_bos 0 \
    --output_name "$OUT" --results_root "$RROOT" \
    > "logs/cb_${OUT}_shard${g}.out" 2>&1 &
  pids+=($!)
done
echo "[$(date '+%F %T')] launched 8 shards: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[$(date '+%F %T')] all shards done (fail=$fail) for $OUT"
$PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$OUT" --results_root "$RROOT"
echo "[$(date '+%F %T')] MERGE_DONE $OUT (shard_fail=$fail)"
