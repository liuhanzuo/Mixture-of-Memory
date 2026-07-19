#!/usr/bin/env bash
# Full 8-GPU sharded held-out PPL run for the OLMo-2 prune-then-heal probe (Paper B).
# EVAL-ONLY (forward-only ppl) on .73. Each config fans out 8 shard processes
# (one per GPU, val windows strided [g::8]), waits, then token-weight-merges.
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
VAL=data/dolmino_now_val.npy
BS=4
mkdir -p logs olmo2_ppl_results
DONE=logs/olmo2_ppl_DONE
rm -f "$DONE"

# config rows: "output_name|base_model|ckpt(empty=full base)"
CONFIGS=(
  "1B_base_full|../models/OLMo-2-0425-1B|"
  "1B_keep7_step50000|../models/OLMo-2-0425-1B|outputs/olmo2_probe2_1B_keep7fresh2_16card/step50000.pt"
  "1B_keep7_step100000|../models/OLMo-2-0425-1B|outputs/olmo2_probe2_1B_keep7fresh2_16card/step100000.pt"
  "1B_keep7_step147000|../models/OLMo-2-0425-1B|outputs/olmo2_probe2_1B_keep7fresh2_16card/step147000.pt"
  "7B_base_full|../models/OLMo-2-1124-7B|"
  "7B_keep10_step10000|../models/OLMo-2-1124-7B|outputs/olmo2_probe2_7B_keep10fresh2/step10000.pt"
)

for row in "${CONFIGS[@]}"; do
  NAME="${row%%|*}"; rest="${row#*|}"
  BASE="${rest%%|*}"; CKPT="${rest#*|}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] CONFIG $NAME base=$BASE ckpt='${CKPT:-<FULL BASE>}'"
  CKARG=""; [ -n "$CKPT" ] && CKARG="--ckpt $CKPT"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" $CKARG \
      --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size $BS \
      --output_name "$NAME" \
      > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] $NAME shards done; merging"
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1
done

echo "[$(date '+%F %T')] ALL CONFIGS DONE" | tee "$DONE"
for row in "${CONFIGS[@]}"; do
  NAME="${row%%|*}"
  echo "--- $NAME ---" >> "$DONE"
  cat "olmo2_ppl_results/${NAME}/summary.json" >> "$DONE" 2>/dev/null
done
echo "[$(date '+%F %T')] wrote $DONE"
