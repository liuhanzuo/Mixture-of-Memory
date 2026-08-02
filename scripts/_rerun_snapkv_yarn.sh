#!/usr/bin/env bash
# PaperA P1.6: SnapKV YaRN 64k/128k rerun (was crashed by rope_theta=None, fixed commit 14bd576).
# LOCAL 8xB200 (wzc1), fixed harness scripts/eval_p16_kvcompress.py.
# SnapKV scheduler was SnapKV-only + SCHED_DONE at 15:42, so yarn cells never auto-rerun -> do manually.
# 3 RULER tasks x {64k,128k} via YaRN, 8 shards each. GPU g runs shard g of all 3 tasks sequentially.
set -u
WD=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$WD" || exit 1
PY=$WD/.venv/bin/python
MODEL=models/Qwen3-8b-local
TASKS=(niah_single_2 niah_multikey_1 variable_tracking)
mkdir -p logs/p16_snapkv_yarn_rerun

echo "[$(date '+%F %T')] ===== SnapKV YaRN 64k/128k rerun start (8xB200) ====="
for g in 0 1 2 3 4 5 6 7; do
(
  for t in "${TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_p16_kvcompress.py --mode ruler --method snapkv \
      --model_path "$MODEL" --device cuda:0 \
      --tasks "$t" --lengths 64k 128k --long_ctx yarn \
      --num_samples 100 --num_shards 8 --shard_index $g \
      --max_capacity_prompt 6657 --window_size 32 \
      --results_folder ruler_results --output_name p16_snapkv_yarn \
      > "logs/p16_snapkv_yarn_rerun/${t}_shard${g}.log" 2>&1
  done
) &
done
wait
echo "[$(date '+%F %T')] ===== SnapKV YaRN rerun ALL DONE ====="
ls ruler_results/p16_snapkv_yarn/ 2>/dev/null | grep -c json
