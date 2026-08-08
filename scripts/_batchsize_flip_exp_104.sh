#!/usr/bin/env bash
# Batch-size flip-cause experiment — extended arms (bs=4 and bs=32)
# Node: .104 (28.83.24.104, 8xH20, zwfy6 disk)
# Arm C: --batch_size 4  -> 7B_shortgpt16_step200000_bs4
# Arm D: --batch_size 32 -> 7B_shortgpt16_step200000_bs32
# If bs hypothesis holds, monotonic increase with bs-pair distance is expected.
set -u
ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/OLMo-2-1124-7B
CKPT=outputs/olmo2_probe2_7B_shortgpt16/step200000.pt
TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

assert_8shards () {
  local D="olmo2_downstream_results/$1"
  local MISS=0
  for g in 0 1 2 3 4 5 6 7; do
    [ -f "$D/shard${g}of8.json" ] || { echo "[SHARD MISSING] $D/shard${g}of8.json"; MISS=$((MISS+1)); }
  done
  [ $MISS -eq 0 ] && return 0
  echo "[ABORT] $MISS/8 shards missing for $1 — NOT merging"
  return 1
}

# prepare datasets once
echo "[$(date '+%F %T')] prepare_data"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
  > logs/batchsize_flip_104_prepare.log 2>&1
tail -3 logs/batchsize_flip_104_prepare.log

for BS in 4 32; do
  NAME="7B_shortgpt16_step200000_bs${BS}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] ARM batch_size=$BS -> $NAME"
  if [ -f "olmo2_downstream_results/$NAME/summary.json" ]; then
    echo "[$(date '+%F %T')] $NAME ALREADY DONE, skipping"
    continue
  fi
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g \
    $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --keep_front_layers 16 --n_fresh_layers 0 \
      --tasks "$TASKS" \
      --num_shards 8 --shard_index $g \
      --batch_size $BS \
      --save_per_example \
      --output_name "$NAME" \
      > "logs/batchsize_flip_104_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] shards done for $NAME; asserting 8/8..."
  assert_8shards "$NAME" || { echo "[FATAL] merge aborted for $NAME"; continue; }
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] summary:"
  cat "olmo2_downstream_results/$NAME/summary.json" 2>/dev/null
  echo
done

echo "[$(date '+%F %T')] ALL DONE"
echo "Results written to:"
echo "  olmo2_downstream_results/7B_shortgpt16_step200000_bs4/summary.json"
echo "  olmo2_downstream_results/7B_shortgpt16_step200000_bs32/summary.json"
