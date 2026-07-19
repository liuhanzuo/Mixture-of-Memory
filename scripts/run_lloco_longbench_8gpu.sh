#!/bin/bash
# LLoCO LongBench eval runner — 3 tasks x 8-way sharded on .73 (8xH20).
# Runs tasks sequentially; within a task, one shard per GPU (0..7); then scores.
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/lloco_env/bin/python
OUT=lloco_results/longbench
mkdir -p "$OUT" logs
export TOKENIZERS_PARALLELISM=false

for TASK in narrativeqa qasper hotpotqa; do
  echo "[runner] $(date) starting $TASK (8 shards)"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_lloco_longbench.py \
      --dataset "$TASK" --num_shards 8 --shard_index "$g" \
      --output_dir "$OUT" >"logs/lloco_${TASK}_g${g}.log" 2>&1 &
  done
  wait
  echo "[runner] $(date) finished $TASK"
done

echo "[runner] $(date) scoring"
$PY scripts/eval_lloco_longbench.py --score_only \
  --datasets narrativeqa qasper hotpotqa --output_dir "$OUT" >logs/lloco_score.log 2>&1
touch logs/lloco_longbench_DONE
echo "[runner] $(date) ALL DONE"
