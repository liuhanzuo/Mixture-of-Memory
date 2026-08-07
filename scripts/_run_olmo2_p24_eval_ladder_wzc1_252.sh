#!/usr/bin/env bash
# Paper B P2.4 / Task #189 — Table 4 ladder pre-SFT re-eval on .252 (wzc1/L20A cc10.0)
#   Purpose: complete the cross-architecture audit for keep8 and keep12 by
#     producing L20A per-item preds for both rungs. Sibling `_v2` H20 batteries
#     already exist on zwfy6 (`_run_olmo2_p24_eval_ladder_prev2_73.sh` output);
#     this driver gives us the wzc1-side twin so `core6_wzc1(X) − core6_zwfy6_v2(X)`
#     can be computed per rung and tested against the "flip count scales with
#     pruning damage" hypothesis (n=2: full base 10 flips; keep14 28 flips).
#   Byte-identical mirror of `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh` —
#     same harnesses, tokenisation, chat_template=False, add_bos=false, bf16
#     autocast, 8-GPU shard+merge, --save_per_example, assert_8shards invariant.
#   Only the arm loop / node / ROOT / output-name suffix change:
#     - 2 rungs (keep8, keep12) instead of 4
#     - ROOT = wzc1 (not zwfy6)
#     - `_wzc1` suffix on output names so we NEVER overwrite the sibling `_v2`
#       H20 summaries on zwfy6 or the Table 4 anchors.
#
# Load-bearing checkpoint provenance (verified with `ls outputs/... 2026-08-08`):
#   keep8       outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt   -> 7B_keep8_step121000_wzc1
#               (48.7 GB; MATCHES the Table 4 headline step for keep8)
#   keep12      outputs/olmo2_probe2_7B_keep12fresh2/step111500.pt  -> 7B_keep12_step111500_wzc1
#               (48.7 GB; DIFFERS from the Table 4 headline `step124000`. On
#                wzc1, step111500 is the ONLY step*.pt in that directory; the
#                zwfy6 sibling used step124000, which does not exist on wzc1.
#                See status/PAPERB_P24_LADDER_WZC1_EVAL.md — this is called out
#                loudly in the report.)
#
# keep10 is intentionally SKIPPED here: the only wzc1-side keep10 checkpoint is
# absent (max step 83500 lives only on zwfy6), and cross-disk scp -O of a
# 48.7 GB file at ~12 MB/s (~70 min) is not worth blocking on. MAIN will decide
# whether to scp+rerun separately.
#
# Output-name convention (all with `_wzc1` suffix on the pre-SFT anchor):
#   PPL              olmo2_ppl_results/<NAME>_wzc1/summary.json
#   core6            olmo2_downstream_results/<NAME>_wzc1/summary.json
#   know5            olmo2_downstream_results/<NAME>_wzc1_know/summary.json
#   MMLU dual        olmo2_mmlu_content_results/<NAME>_wzc1/summary.json
#   closedbook       olmo2_closedbook_results/<NAME>_wzc1/summary.json
#
# Per-item predictions retained (downstream --save_per_example; MMLU-content and
# closedbook harnesses write per-item files by default) — required for
# per-item McNemar / paired bootstrap vs zwfy6_v2.
#
# Launch (on .252, cd to wzc1 root):
#   setsid nohup bash scripts/_run_olmo2_p24_eval_ladder_wzc1_252.sh \
#     > logs/p24_eval_ladder_wzc1_252.log 2>&1 &
set -u

ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/OLMo-2-1124-7B
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

# Two rungs. Arrays index-aligned.
ARM_NAMES=(keep8 keep12)
CKPTS=(
  "outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt"
  "outputs/olmo2_probe2_7B_keep12fresh2/step111500.pt"
)
NAMES=(
  "7B_keep8_step121000_wzc1"
  "7B_keep12_step111500_wzc1"
)

# ---- pre-flight ----
for CK in "${CKPTS[@]}"; do
  if [ ! -f "$CK" ]; then
    echo "[$(date '+%F %T')] FATAL: ckpt not found: $CK"; exit 2
  fi
done
if [ ! -d "$BASE" ]; then
  echo "[$(date '+%F %T')] FATAL: base model dir not found: $BASE"; exit 2
fi
if [ ! -f "$VAL" ]; then
  echo "[$(date '+%F %T')] FATAL: val_path not found: $VAL"; exit 2
fi
echo "[$(date '+%F %T')] pre-flight OK: 2/2 ckpts present; BASE and VAL present"

# assert_8shards <results_root> <NAME> <shard_pat>
#   Confirms all 8 shard files exist; if not, ABORT the merge (partial merge =
#   silent contamination). Called before every merge step.
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
    echo "[$(date '+%F %T')] ABORT merge for $NAME: $MISS/8 shards missing (partial merge would silently contaminate results). Rerun the failed shards."
    return 1
  fi
  return 0
}

# ---------------- eval helpers (8-GPU shard+merge; --save_per_example on MC) ----

run_ppl () {
  local NAME=$1 CKPT=$2
  local D="olmo2_ppl_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] PPL $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- PPL $NAME (ckpt=$CKPT) ---"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
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
  echo "[$(date '+%F %T')] --- downstream $NAME (ckpt=$CKPT tasks=$TASKS) ---"
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
  echo "[$(date '+%F %T')] --- mmlu-content $NAME (ckpt=$CKPT) ---"
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
  echo "[$(date '+%F %T')] --- closedbook $NAME (ckpt=$CKPT) ---"
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

echo "[$(date '+%F %T')] ======== P2.4 LADDER PRE-SFT _wzc1 EVAL BATTERY (.252, L20A) START ========"
echo "[$(date '+%F %T')] Rungs: ${ARM_NAMES[*]}"
echo "[$(date '+%F %T')] Ckpts: ${CKPTS[*]}"
echo "[$(date '+%F %T')] Names: ${NAMES[*]}"
echo "[$(date '+%F %T')] NOTE: keep12 uses step111500 (only step*.pt on wzc1); the zwfy6 sibling used step124000. This is called out in status/PAPERB_P24_LADDER_WZC1_EVAL.md."

# Serial across arms; parallel across 8 shards within each battery.
# 2 arms × 5 harnesses = 10 harness invocations ≈ 2 h total on L20A.
for i in 0 1; do
  ARM="${ARM_NAMES[$i]}"
  CKPT="${CKPTS[$i]}"
  NAME="${NAMES[$i]}"
  echo "[$(date '+%F %T')] ================ $ARM ($NAME) ================"
  echo "[$(date '+%F %T')] ckpt=$CKPT"
  run_ppl        "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $ARM PPL FAILED, continuing"
  run_downstream "$NAME"        "$CKPT" "$CORE" || echo "[$(date '+%F %T')] $ARM core6 FAILED"
  run_downstream "${NAME}_know" "$CKPT" "$KNOW" || echo "[$(date '+%F %T')] $ARM know5 FAILED"
  run_mmlu_content "$NAME"      "$CKPT" || echo "[$(date '+%F %T')] $ARM mmlu-content FAILED"
  run_closedbook "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $ARM closedbook FAILED"
  echo "[$(date '+%F %T')] ================ $ARM DONE ================"
done

echo "[$(date '+%F %T')] ======== P2.4 LADDER PRE-SFT _wzc1 EVAL BATTERY (.252) DONE ========"
echo "[$(date '+%F %T')] Outputs:"
for i in 0 1; do
  NAME="${NAMES[$i]}"
  echo "  ${ARM_NAMES[$i]}:"
  echo "    PPL:            olmo2_ppl_results/$NAME/summary.json"
  echo "    core6+know5:    olmo2_downstream_results/${NAME}{,_know}/summary.json"
  echo "    MMLU dual:      olmo2_mmlu_content_results/$NAME/summary.json"
  echo "    closedbook:     olmo2_closedbook_results/$NAME/summary.json"
done
