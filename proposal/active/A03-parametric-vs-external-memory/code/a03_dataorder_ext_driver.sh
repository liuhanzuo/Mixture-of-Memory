#!/usr/bin/env bash
# A03 data-order replication (seed 43/44/45) -- eval driver.
#
# Mirrors code/a03_arm6_ext_driver.sh EXACTLY in harness, sharding and result-dir
# convention. Two deliberate differences:
#
#  1. ONLY step220000 is evaluated. The 5k intermediates are NOT scored. These
#     runs exist to answer one pre-registered question at one step
#     (DATAORDER_PREREG.md); scoring step205/210/215 would add 9 more
#     arm-axis cells whose only use would be post-hoc re-reading of an
#     oscillation we have already characterised. Do not "just also run" them.
#  2. SEEDS is a whitelist. The zwfy6 disk is shared by .73/.82/.104, so every
#     seed's output_dir is visible from every node. Each node's watcher passes
#     ONLY its own seed, otherwise two nodes race on the same result dir.
#
# Result dirs (must stay in sync with recompute_cpt_trajectory_paired.py ARMS):
#     olmo2_mmlu_content_results/A03_1B_dataorder_seed<S>_step220000
#     olmo2_closedbook_results/  A03_1B_dataorder_seed<S>_step220000      (popqa,triviaqa)
#     olmo2_closedbook_results/  A03_1B_dataorder_seed<S>_step220000_nq   (nq_open)
# Baseline for the paired diff is A03_1B_keep7_step200k -- the SAME baseline as
# Arm 3 / Arm 4 / Arm 6. Do not invent a new one.
#
# Usage: SEEDS=43 bash a03_dataorder_ext_driver.sh
set -u
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W || exit 3
PY=/opt/conda/envs/torch-base/bin/python
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
SEEDS="${SEEDS:-43 44 45}"
STEP=220000
export HF_DATASETS_CACHE=$W/data/hf_datasets_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
unset http_proxy https_proxy all_proxy

for S in $SEEDS; do
  PROG=logs/a03_dataorder_seed${S}_eval_progress.log
  note() { printf "[%s] ext-drv(seed%s): %s\n" "$(date "+%m-%d %H:%M:%S")" "$S" "$*" | tee -a $PROG; }

  CK=outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed${S}/step${STEP}.pt
  [ -f "$CK" ] || { note "[skip] step${STEP} no ckpt"; continue; }

  # GPUs must be free. Checked per seed, not once up front: the trainer for THIS
  # seed may still be shutting down when the watcher fires.
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}")
  if [ "$used" -gt 8000 ]; then note "REFUSE: ${used}MiB held"; continue; fi

  # Internal torch.load probe: independent of the watcher's size guard, so a
  # partial write is bounced even if the size heuristic is fooled.
  header_ok=$($PY -c "import torch; torch.load(\"$CK\", map_location=\"cpu\", weights_only=False); print(\"ok\")" 2>/dev/null || echo bad)
  if [ "$header_ok" != "ok" ]; then note "[skip] step${STEP} not readable (partial write)"; continue; fi

  TAG=A03_1B_dataorder_seed${S}_step${STEP}
  if [ ! -f "olmo2_mmlu_content_results/$TAG/summary.json" ]; then
    note "MMLU START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py --base_model $BASE --ckpt $CK --keep_front_layers 7 --n_fresh_layers 2 --output_name $TAG --num_shards 8 --shard_index $g --batch_size 16 --content_desc full > logs/${TAG}_mmlu_shard${g}.log 2>&1 &
    done
    wait
    n=$(ls olmo2_mmlu_content_results/$TAG/per_example_mmlu_shard*of8.jsonl 2>/dev/null | wc -l)
    if [ "$n" -ne 8 ]; then note "ABORT MMLU: $n/8 shards"; rm -rf olmo2_mmlu_content_results/$TAG; continue; fi
    $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name $TAG --num_shards 8 >> logs/${TAG}_mmlu_merge.log 2>&1 || true
    note "MMLU DONE ($n/8)"
  fi

  if [ ! -f "olmo2_closedbook_results/$TAG/summary.json" ]; then
    note "CB(pt) START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py --base_model $BASE --ckpt $CK --keep_front_layers 7 --n_fresh_layers 2 --tasks popqa,triviaqa --output_name $TAG --num_shards 8 --shard_index $g > logs/${TAG}_cb_shard${g}.log 2>&1 &
    done
    wait
    for t in popqa triviaqa; do
      n=$(ls olmo2_closedbook_results/$TAG/per_example_${t}_shard*of8.jsonl 2>/dev/null | wc -l)
      [ "$n" -eq 8 ] || note "WARN CB(pt) $t only $n/8 shards -- recompute will hard-fail this cell"
    done
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $TAG --num_shards 8 >> logs/${TAG}_cb_merge.log 2>&1 || true
    note "CB(pt) DONE"
  fi

  NQTAG=${TAG}_nq
  if [ ! -f "olmo2_closedbook_results/$NQTAG/summary.json" ]; then
    note "CB(nq) START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py --base_model $BASE --ckpt $CK --keep_front_layers 7 --n_fresh_layers 2 --tasks nq_open --output_name $NQTAG --num_shards 8 --shard_index $g > logs/${NQTAG}_shard${g}.log 2>&1 &
    done
    wait
    n=$(ls olmo2_closedbook_results/$NQTAG/per_example_nq_open_shard*of8.jsonl 2>/dev/null | wc -l)
    [ "$n" -eq 8 ] || note "WARN CB(nq) only $n/8 shards -- recompute will hard-fail this cell"
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $NQTAG --num_shards 8 >> logs/${NQTAG}_merge.log 2>&1 || true
    note "CB(nq) DONE"
  fi
  note "all 4 axes done for seed$S step${STEP}"
done
printf "[%s] ext-drv exit (SEEDS=%s)\n" "$(date "+%m-%d %H:%M:%S")" "$SEEDS"
