#!/bin/bash
# Eval one unfreeze-sweep checkpoint: fullchain oracle qa5 16k, n100, sharded on
# the given GPUs. fullchain oracle = true supporting-fact chunk 100% in the
# reforward window (selection perfect) → pure READOUT probe. Args:
#   $1 = arm tag (e.g. arm2_top16_s400)
#   $2 = checkpoint .pt path (relative to repo root)
#   $3 = adapter_config.json path
#   $4 = comma-separated GPU ids to shard over (e.g. 0,1,2,3)
set -euo pipefail
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
TAG="$1"; CKPT="$2"; ACFG="$3"; GPUS="$4"
OUT="babilong_results/unfreeze_${TAG}_fc16k"
mkdir -p "$OUT" logs/unfreeze_eval
IFS=',' read -ra GARR <<< "$GPUS"
NSH=${#GARR[@]}
echo "[eval] $TAG ckpt=$CKPT nshards=$NSH gpus=$GPUS -> $OUT"
for i in "${!GARR[@]}"; do
  g=${GARR[$i]}
  CUDA_VISIBLE_DEVICES=$g setsid "$R/.venv/bin/python" -u scripts/probe_fullchain_oracle_qa5.py \
    --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --tasks qa5 --lengths 16k --limit 100 --num_shards "$NSH" --shard_idx "$i" \
    --oracle_mode fullchain --device cuda:0 \
    --results_folder "$OUT" \
    > "logs/unfreeze_eval/${TAG}_shard${i}.log" 2>&1 &
  echo "  shard$i -> GPU$g PID $!"
done
wait
echo "[eval] $TAG done. CSVs in $OUT"
