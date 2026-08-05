#!/usr/bin/env bash
# Re-run ONLY the shards that died with CUBLAS_STATUS_ALLOC_FAILED on 2026-08-05,
# restoring reheal_step{55000,57500} to the canonical 8-shard/4096-window basis
# so they are comparable with step30000..52500 when bracketing CROSS2=11.4983.
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/OLMo-2-1124-7B
VAL=data/dolmino_now_val.npy
RROOT=.t27_tmp/t103_matchedppl/ppl_results
for st in 55000 57500; do
  NAME="reheal_step${st}"
  CKPT="outputs/olmo2_keep14_densesave_reheal/step${st}.pt"
  echo "=== [$(date '+%F %T')] refill $NAME shards 0,1,7 ==="
  for g in 0 1 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --val_path "$VAL" --num_shards 8 --shard_index "$g" --batch_size 4 \
      --results_root "$RROOT" --output_name "$NAME" \
      > "logs/t103ppl_${NAME}_shard${g}.refill.log" 2>&1 &
  done
  wait
  echo "--- merge $NAME (8/8 校验生效) ---"
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --results_root "$RROOT" --output_name "$NAME" 2>&1
done
echo "[$(date '+%F %T')] REFILL DONE"
