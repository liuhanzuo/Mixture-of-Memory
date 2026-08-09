#!/usr/bin/env bash
# A03 Arm 6 (mid-low-LR CPT) — eval driver, mirrors Arm 4 ext-drv
# Score step205/210/215/220 vs Arm 2 baseline (same 4 A03 axes).
set -u
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/OLMo-2-0425-1B
PROG=logs/a03_arm6_trajectory_progress.log
export HF_DATASETS_CACHE=$W/data/hf_datasets_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
unset http_proxy https_proxy all_proxy
note() { printf "[%s] ext-drv: %s\n" "$(date "+%m-%d %H:%M:%S")" "$*" | tee -a $PROG; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}")
if [ "$used" -gt 8000 ]; then note "REFUSE: ${used}MiB held"; exit 0; fi

for STEP in 205000 210000 215000 220000; do
  CK=outputs/olmo2_probe2_1B_keep7f2_dolmino_arm6_lowerband20k/step${STEP}.pt
  [ -f "$CK" ] || { note "[skip] step${STEP} no ckpt"; continue; }
  header_ok=$($PY -c "import torch; torch.load(\"$CK\", map_location=\"cpu\", weights_only=False); print(\"ok\")" 2>/dev/null || echo bad)
  if [ "$header_ok" != "ok" ]; then note "[skip] step${STEP} not readable (partial write)"; continue; fi

  TAG=A03_1B_arm6_lowerband_step${STEP}
  if [ ! -f "olmo2_mmlu_content_results/$TAG/summary.json" ]; then
    note "MMLU step${STEP} START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py --base_model $BASE --ckpt $CK --keep_front_layers 7 --n_fresh_layers 2 --output_name $TAG --num_shards 8 --shard_index $g --batch_size 16 --content_desc full > logs/${TAG}_mmlu_shard${g}.log 2>&1 &
    done
    wait
    n=$(ls olmo2_mmlu_content_results/$TAG/per_example_mmlu_shard*of8.jsonl 2>/dev/null | wc -l)
    if [ "$n" -ne 8 ]; then note "ABORT MMLU step${STEP}: $n/8 shards"; rm -rf olmo2_mmlu_content_results/$TAG; continue; fi
    $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name $TAG --num_shards 8 >> logs/${TAG}_mmlu_merge.log 2>&1 || true
    note "MMLU step${STEP} DONE ($n/8)"
  fi
  if [ ! -f "olmo2_closedbook_results/$TAG/summary.json" ]; then
    note "CB(pt) step${STEP} START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py --base_model $BASE --ckpt $CK --keep_front_layers 7 --n_fresh_layers 2 --tasks popqa,triviaqa --output_name $TAG --num_shards 8 --shard_index $g > logs/${TAG}_cb_shard${g}.log 2>&1 &
    done
    wait
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $TAG --num_shards 8 >> logs/${TAG}_cb_merge.log 2>&1 || true
    note "CB(pt) step${STEP} DONE"
  fi
  NQTAG=A03_1B_arm6_lowerband_step${STEP}_nq
  if [ ! -f "olmo2_closedbook_results/$NQTAG/summary.json" ]; then
    note "CB(nq) step${STEP} START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py --base_model $BASE --ckpt $CK --keep_front_layers 7 --n_fresh_layers 2 --tasks nq_open --output_name $NQTAG --num_shards 8 --shard_index $g > logs/${NQTAG}_shard${g}.log 2>&1 &
    done
    wait
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $NQTAG --num_shards 8 >> logs/${NQTAG}_merge.log 2>&1 || true
    note "CB(nq) step${STEP} DONE"
  fi
done
note "ext-drv exit"
