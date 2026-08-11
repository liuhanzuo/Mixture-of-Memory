#!/usr/bin/env bash
# A04 Pilot One Stage B -- eval driver.
#
# Mirrors code/a03_dataorder_ext_driver.sh (which fired cleanly for seeds
# 43/44 on 2026-08-11 04:23-04:29). Deliberate differences:
#
#  1. ARM: keep12+fresh2 (not keep7+fresh2). --keep_front_layers 12 --n_fresh_layers 2.
#     The pruned model has 14/16 layers = 87.5% depth (vs Pilot Zero's 56.2%).
#  2. STEP: 5000 (not 220000). Per PILOT_ONE_PREREG.md 2ac0b5a.
#  3. SEED whitelist: 101|102|103 (pre-registered, not 43|44|45).
#  4. Result-dir tag: A04_1B_stageB_keep12_seed<S>_step5000 -- distinct namespace
#     from A03 so the Stage-A driver's dir_template picks up the right shards.
#  5. Ckpt dir: outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed<S>/ (matches
#     scripts/_run_a04_stageB.sh's --output_dir).
#
# Baseline for paired analyses is NOT the same as A03. Stage-A's K2 rule uses
# the *unpaired* seed-vs-seed spread; if downstream analyses want paired-vs-base,
# they must decide their own base. This driver does not encode one.
#
# Usage: SEEDS=101 bash scripts/_run_a04_stageB_ext_driver.sh
set -u
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W || exit 3
PY=/opt/conda/envs/torch-base/bin/python
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
SEEDS="${SEEDS:?SEEDS whitelist must be set (subset of 101 102 103)}"
STEP=5000
KEEP=12
FRESH=2
export HF_DATASETS_CACHE=$W/data/hf_datasets_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
unset http_proxy https_proxy all_proxy

for S in $SEEDS; do
  case "$S" in 101|102|103) ;; *)
    echo "FATAL seed=$S outside pre-registered whitelist {101,102,103}"; continue ;;
  esac
  PROG=logs/a04_stageB_seed${S}_eval_progress.log
  note() { printf "[%s] a04-ext(seed%s): %s\n" "$(date "+%m-%d %H:%M:%S")" "$S" "$*" | tee -a $PROG; }

  CK=outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed${S}/step${STEP}.pt
  [ -f "$CK" ] || { note "[skip] step${STEP} no ckpt at $CK"; continue; }

  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}")
  if [ "$used" -gt 8000 ]; then note "REFUSE: ${used}MiB held"; continue; fi

  header_ok=$($PY -c "import torch; torch.load(\"$CK\", map_location=\"cpu\", weights_only=False); print(\"ok\")" 2>/dev/null || echo bad)
  if [ "$header_ok" != "ok" ]; then note "[skip] step${STEP} not readable (partial write)"; continue; fi

  TAG=A04_1B_stageB_keep12_seed${S}_step${STEP}
  if [ ! -f "olmo2_mmlu_content_results/$TAG/summary.json" ]; then
    note "MMLU START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py --base_model $BASE --ckpt $CK --keep_front_layers $KEEP --n_fresh_layers $FRESH --output_name $TAG --num_shards 8 --shard_index $g --batch_size 16 --content_desc full > logs/${TAG}_mmlu_shard${g}.log 2>&1 &
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
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py --base_model $BASE --ckpt $CK --keep_front_layers $KEEP --n_fresh_layers $FRESH --tasks popqa,triviaqa --output_name $TAG --num_shards 8 --shard_index $g > logs/${TAG}_cb_shard${g}.log 2>&1 &
    done
    wait
    for t in popqa triviaqa; do
      n=$(ls olmo2_closedbook_results/$TAG/per_example_${t}_shard*of8.jsonl 2>/dev/null | wc -l)
      [ "$n" -eq 8 ] || note "WARN CB(pt) $t only $n/8 shards -- Stage-A driver will hard-fail this cell"
    done
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $TAG --num_shards 8 >> logs/${TAG}_cb_merge.log 2>&1 || true
    note "CB(pt) DONE"
  fi

  NQTAG=${TAG}_nq
  if [ ! -f "olmo2_closedbook_results/$NQTAG/summary.json" ]; then
    note "CB(nq) START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py --base_model $BASE --ckpt $CK --keep_front_layers $KEEP --n_fresh_layers $FRESH --tasks nq_open --output_name $NQTAG --num_shards 8 --shard_index $g > logs/${NQTAG}_shard${g}.log 2>&1 &
    done
    wait
    n=$(ls olmo2_closedbook_results/$NQTAG/per_example_nq_open_shard*of8.jsonl 2>/dev/null | wc -l)
    [ "$n" -eq 8 ] || note "WARN CB(nq) only $n/8 shards -- Stage-A driver will hard-fail this cell"
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $NQTAG --num_shards 8 >> logs/${NQTAG}_merge.log 2>&1 || true
    note "CB(nq) DONE"
  fi
  note "all 4 axes done for seed$S step${STEP}"
done
printf "[%s] a04-ext exit (SEEDS=%s)\n" "$(date "+%m-%d %H:%M:%S")" "$SEEDS"
