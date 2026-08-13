#!/usr/bin/env bash
# A04 SHALLOW RUNG LADDER -- 4-axis eval driver for the two new 1B arms.
#
# Mirrors scripts/_run_a04_stageB_ext_driver.sh (which produced every Stage B
# capability number) with exactly three deliberate differences:
#
#  1. ARMS: keep{13,14}+fresh2 instead of keep12+fresh2. The eval harness must be
#     told the ARM'S OWN keep/fresh (it rebuilds the pruned shell before loading
#     the state dict), so KEEP is derived from the arm, never hardcoded to 12.
#  2. Result-dir tag: A04_1B_shallow_keep<K>_seed<S>_step5000 -- a distinct
#     namespace from Stage B's A04_1B_stageB_keep12_seed<S>_step5000.
#  3. Ckpt dir: outputs/olmo2_probe2_1B_keep<K>f2_dolmino_shallow_seed<S>/
#     (matches scripts/_run_a04_shallow_ladder.sh's --output_dir).
#
# EVERYTHING ELSE IS BYTE-FOR-BYTE THE STAGE B PROTOCOL: same two eval scripts,
# same --content_desc full, same closed-book bs (harness default 32), same mmlu
# --batch_size 16, same 8-shard split, same nq_open in its own _nq tag, same
# offline HF env, same merge step. If any of these drifted the new arms would not
# be comparable to keep12 and the whole ladder would be void.
#
# SHARD COMPLETENESS IS HARD-CHECKED HERE AND AGAIN IN THE ANALYSIS. This driver
# aborts (and removes the partial dir) rather than merge 5-of-8: a silently
# merged partial shard set has corrupted results in this repo before. The
# analysis re-asserts the index SET == {0..7}, exact item counts, 0 duplicate
# item_id and 0 nan, because a driver check can be bypassed by a later caller.
#
# NODE BUDGET (2026-08-13 dispatch, hard):
#   ALLOWED  : .73 (28.85.35.73), .82 (28.82.250.82) -- 8xH20, zwfy6
#   FORBIDDEN: .104 (paperC Qwen3-8B heal), .21 + LOCAL (SparseForge #246)
# Enforced below by IP, not by trusting the caller.
#
# Usage (ON the target node):  ARMS=14 bash code/a04_shallow_ladder_eval_driver.sh
#                              ARMS="13 14" bash ...   (if both ckpts are local)
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
ARMS="${ARMS:?ARMS must be set (subset of '13 14')}"
SEED="${SEED:-101}"
STEP=5000
FRESH=2

export HF_DATASETS_CACHE=$W/data/hf_datasets_cache HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
unset http_proxy https_proxy all_proxy

# ---- forbidden-node guard, by IP -------------------------------------------
HOSTIP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^28\.' | head -1)
echo "[guard] host ip (28.x) = ${HOSTIP:-none}"
case "$HOSTIP" in
  28.83.24.104) echo "[guard] REFUSING: .104 runs paperC Qwen3 heal"; exit 11 ;;
  28.89.19.21)  echo "[guard] REFUSING: .21 runs SparseForge #246";  exit 11 ;;
esac

for K in $ARMS; do
  case "$K" in 13|14) ;; *)
    echo "FATAL KEEP=$K outside the pre-registered shallow set {13,14}"; continue ;;
  esac
  PROG=logs/a04_shallow_keep${K}_seed${SEED}_eval_progress.log
  note() { printf "[%s] a04-shallow(keep%s): %s\n" "$(date "+%m-%d %H:%M:%S")" "$K" "$*" | tee -a $PROG; }

  CK=outputs/olmo2_probe2_1B_keep${K}f2_dolmino_shallow_seed${SEED}/step${STEP}.pt
  [ -f "$CK" ] || { note "[skip] no ckpt at $CK"; continue; }

  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}")
  if [ "$used" -gt 8000 ]; then note "REFUSE: ${used}MiB held"; continue; fi

  # ckpt must be readable AND carry the arch we think it does -- an eval that
  # rebuilds the WRONG shell would load strict=False-ish garbage or crash late.
  META=$("$PY" - "$CK" <<'PYEOF' 2>/dev/null || echo bad
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"{d.get('keep_front_layers')} {d.get('n_fresh_layers')} "
      f"{d.get('num_hidden_layers')} {d.get('step')} {d.get('seed')}")
PYEOF
)
  if [ "$META" = "bad" ]; then note "[skip] ckpt not readable (partial write)"; continue; fi
  set -- $META
  if [ "$1" != "$K" ] || [ "$2" != "$FRESH" ] || [ "$3" != "$((K+FRESH))" ] \
     || [ "$4" != "$STEP" ] || [ "$5" != "$SEED" ]; then
    note "FATAL ckpt arch/step/seed mismatch: got keep=$1 fresh=$2 depth=$3 step=$4 seed=$5, expected keep=$K fresh=$FRESH depth=$((K+FRESH)) step=$STEP seed=$SEED"
    continue
  fi
  note "ckpt PREFLIGHT-ASSERT OK: keep=$1 fresh=$2 depth=$3 step=$4 seed=$5"

  TAG=A04_1B_shallow_keep${K}_seed${SEED}_step${STEP}
  note "DRIVER START $TAG mmlu_bs=16 cb_bs=32(harness default) content_desc=full"

  if [ ! -f "olmo2_mmlu_content_results/$TAG/summary.json" ]; then
    note "MMLU START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
        --base_model $BASE --ckpt $CK \
        --keep_front_layers $K --n_fresh_layers $FRESH \
        --output_name $TAG --num_shards 8 --shard_index $g \
        --batch_size 16 --content_desc full \
        > logs/${TAG}_mmlu_shard${g}.log 2>&1 &
    done
    wait
    n=$(ls olmo2_mmlu_content_results/$TAG/per_example_mmlu_shard*of8.jsonl 2>/dev/null | wc -l)
    if [ "$n" -ne 8 ]; then note "ABORT MMLU: $n/8 shards -- removing partial dir"; rm -rf olmo2_mmlu_content_results/$TAG; continue; fi
    $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name $TAG --num_shards 8 >> logs/${TAG}_mmlu_merge.log 2>&1 || true
    note "MMLU DONE ($n/8)"
  else
    note "MMLU already present, skipping"
  fi

  if [ ! -f "olmo2_closedbook_results/$TAG/summary.json" ]; then
    note "CB(pt) START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
        --base_model $BASE --ckpt $CK \
        --keep_front_layers $K --n_fresh_layers $FRESH \
        --tasks popqa,triviaqa --output_name $TAG --num_shards 8 --shard_index $g \
        > logs/${TAG}_cb_shard${g}.log 2>&1 &
    done
    wait
    BAD=0
    for t in popqa triviaqa; do
      n=$(ls olmo2_closedbook_results/$TAG/per_example_${t}_shard*of8.jsonl 2>/dev/null | wc -l)
      [ "$n" -eq 8 ] || { note "ABORT CB(pt): $t only $n/8 shards"; BAD=1; }
    done
    if [ "$BAD" -ne 0 ]; then note "removing partial dir"; rm -rf olmo2_closedbook_results/$TAG; continue; fi
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $TAG --num_shards 8 >> logs/${TAG}_cb_merge.log 2>&1 || true
    note "CB(pt) DONE (8/8 both tasks)"
  else
    note "CB(pt) already present, skipping"
  fi

  NQTAG=${TAG}_nq
  if [ ! -f "olmo2_closedbook_results/$NQTAG/summary.json" ]; then
    note "CB(nq) START"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
        --base_model $BASE --ckpt $CK \
        --keep_front_layers $K --n_fresh_layers $FRESH \
        --tasks nq_open --output_name $NQTAG --num_shards 8 --shard_index $g \
        > logs/${NQTAG}_shard${g}.log 2>&1 &
    done
    wait
    n=$(ls olmo2_closedbook_results/$NQTAG/per_example_nq_open_shard*of8.jsonl 2>/dev/null | wc -l)
    if [ "$n" -ne 8 ]; then note "ABORT CB(nq): $n/8 shards -- removing partial dir"; rm -rf olmo2_closedbook_results/$NQTAG; continue; fi
    $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name $NQTAG --num_shards 8 >> logs/${NQTAG}_merge.log 2>&1 || true
    note "CB(nq) DONE (8/8)"
  else
    note "CB(nq) already present, skipping"
  fi
  note "all 4 axes done for keep${K} seed${SEED} step${STEP}"
done
printf "[%s] a04-shallow eval driver exit (ARMS=%s)\n" "$(date "+%m-%d %H:%M:%S")" "$ARMS"
