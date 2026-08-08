#!/usr/bin/env bash
# Paper B — within-disk floor variance controls (v3 batch, 2026-08-08)
#   Purpose: extend PAPERB_WITHIN_DISK_FLOOR (n=2 comparisons on three rungs) to
#     n=3 by adding one more same-disk, same-architecture repeat per rung, and
#     to cover rungs that currently have n=1 (keep14, full32-base-wzc1).
#   Byte-identical mirror of `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh` —
#     same harnesses, tokenisation, chat_template=False, add_bos=false, bf16
#     autocast, 8-GPU shard+merge, --save_per_example, assert_8shards invariant.
#   New `_v3` (or `_v2` for keep14 / full32-base-wzc1) output names, so no
#   pre-existing summary.json is touched.
#
# ARM plan (one arm per node, ARM= first arg):
#   .73   -> ARM=keep14         zwfy6/H20 cc9.0 keep14 step200000 -> 7B_keep14_step200000_v2
#   .82   -> ARM=keep8          zwfy6/H20 cc9.0 keep8  step121000 -> 7B_keep8_step121000_v3
#   .104  -> ARM=shortgpt16     zwfy6/H20 cc9.0 sg16   step200000 -> 7B_shortgpt16_step200000_v3
#   .252  -> ARM=full32_base    wzc1/L20A cc10.0 HF base (no ckpt) -> 7B_full32_base_wzc1_v2
#
# Each battery: PPL + core6 + know5 + MMLU letter+content + closedbook (popqa+triviaqa).
# `--save_per_example` retained on downstream/MMLU/closedbook so per-item preds land on disk.
#
# Launch (per node):
#   setsid nohup bash scripts/_run_olmo2_within_disk_floor_v3.sh <ARM> \
#     > logs/within_disk_floor_v3_<ARM>.log 2>&1 &
set -u

ARM="${1:?Usage: $0 ARM (one of: keep14 keep8 shortgpt16 full32_base)}"

# Node-aware ROOT (wzc1 for full32_base on .252; zwfy6 for the three H20s).
case "$ARM" in
  full32_base)
    ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
    BASE=../models/OLMo-2-1124-7B
    CKPT=""                         # HF base only, no --ckpt
    NAME="7B_full32_base_wzc1_v2"
    ;;
  keep14)
    ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
    BASE=../models/OLMo-2-1124-7B
    CKPT="outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt"
    NAME="7B_keep14_step200000_v2"
    ;;
  keep8)
    ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
    BASE=../models/OLMo-2-1124-7B
    CKPT="outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt"
    NAME="7B_keep8_step121000_v3"
    ;;
  shortgpt16)
    ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
    BASE=../models/OLMo-2-1124-7B
    CKPT="outputs/olmo2_probe2_7B_shortgpt16/step200000.pt"
    NAME="7B_shortgpt16_step200000_v3"
    ;;
  *)
    echo "unknown ARM=$ARM"; exit 2 ;;
esac

cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
VAL=data/dolmino_now_val.npy

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs olmo2_ppl_results olmo2_downstream_results \
         olmo2_mmlu_content_results olmo2_closedbook_results "$HF_DATASETS_CACHE"

CORE="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

# ---- pre-flight ----
if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then
  echo "[$(date '+%F %T')] FATAL: ckpt not found: $CKPT"; exit 2
fi
if [ ! -d "$BASE" ]; then
  echo "[$(date '+%F %T')] FATAL: base_model not found: $BASE"; exit 2
fi
echo "[$(date '+%F %T')] ARM=$ARM  NAME=$NAME  CKPT=${CKPT:-<HF base only>}  ROOT=$ROOT"
echo "[$(date '+%F %T')] pre-flight OK"

# assert_8shards <results_root> <NAME> <shard_pat>
#   Confirms all 8 shard files exist; if not, ABORT merge (partial merge = silent contamination).
assert_8shards () {
  local RROOT=$1 NAME=$2 PAT=$3
  local D="$RROOT/$NAME"
  local MISS=0
  for g in 0 1 2 3 4 5 6 7; do
    if ! ls "$D"/${PAT/\{g\}/$g} >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] SHARD MISSING: $D/${PAT/\{g\}/$g}"
      MISS=$((MISS+1))
    fi
  done
  if [ $MISS -gt 0 ]; then
    echo "[$(date '+%F %T')] ABORT merge for $NAME: $MISS/8 shards missing"
    return 1
  fi
  return 0
}

run_ppl () {
  local NAME=$1 CKPT=$2
  local D="olmo2_ppl_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] PPL $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- PPL $NAME (ckpt=${CKPT:-<base>}) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" "${CKARG[@]}" \
      --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
      --output_name "$NAME" \
      > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_ppl_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] PPL summary ($NAME):"; cat "$D/summary.json" 2>/dev/null; echo
}

run_downstream () {
  local NAME=$1 CKPT=$2 TASKS=$3
  local D="olmo2_downstream_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] downstream $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- downstream $NAME (ckpt=${CKPT:-<base>} tasks=$TASKS) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
    > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1 || true
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" "${CKARG[@]}" --tasks "$TASKS" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --save_per_example \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_downstream_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] downstream summary ($NAME):"; cat "$D/summary.json" 2>/dev/null | head -40; echo
}

run_mmlu_content () {
  local NAME=$1 CKPT=$2
  local D="olmo2_mmlu_content_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] mmlu-content $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- mmlu-content $NAME (ckpt=${CKPT:-<base>}) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data \
    > "logs/olmo2_mmluc_${NAME}_prepare.log" 2>&1 || true
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" "${CKARG[@]}" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAME" \
      > "logs/olmo2_mmluc_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_mmlu_content_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] mmlu-content summary ($NAME):"; cat "$D/summary.json" 2>/dev/null | head -40; echo
}

run_closedbook () {
  local NAME=$1 CKPT=$2
  local D="olmo2_closedbook_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] closedbook $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- closedbook $NAME (ckpt=${CKPT:-<base>}) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  $PY scripts/eval_olmo2_closedbook_qa.py --prepare_data --tasks popqa,triviaqa \
    > "logs/olmo2_cb_${NAME}_prepare.log" 2>&1 || true
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" "${CKARG[@]}" --tasks popqa,triviaqa \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAME" \
      > "logs/olmo2_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_closedbook_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] closedbook summary ($NAME):"; cat "$D/summary.json" 2>/dev/null | head -40; echo
}

echo "[$(date '+%F %T')] ======== WITHIN-DISK FLOOR v3 EVAL BATTERY START ($ARM -> $NAME) ========"

run_ppl          "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $ARM PPL FAILED, continuing"
run_downstream   "$NAME"        "$CKPT" "$CORE" || echo "[$(date '+%F %T')] $ARM core6 FAILED"
run_downstream   "${NAME}_know" "$CKPT" "$KNOW" || echo "[$(date '+%F %T')] $ARM know5 FAILED"
run_mmlu_content "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $ARM mmlu-content FAILED"
run_closedbook   "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $ARM closedbook FAILED"

echo "[$(date '+%F %T')] ======== WITHIN-DISK FLOOR v3 EVAL BATTERY DONE ($ARM -> $NAME) ========"
echo "[$(date '+%F %T')] Outputs:"
echo "  PPL:            olmo2_ppl_results/$NAME/summary.json"
echo "  core6+know5:    olmo2_downstream_results/${NAME}{,_know}/summary.json"
echo "  MMLU dual:      olmo2_mmlu_content_results/$NAME/summary.json"
echo "  closedbook:     olmo2_closedbook_results/$NAME/summary.json"
