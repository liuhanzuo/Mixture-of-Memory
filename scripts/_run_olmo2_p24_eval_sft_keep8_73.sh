#!/usr/bin/env bash
# Paper B P2.4 post-SFT eval battery for the keep8fresh2 arm on .73 (diskB / zwfy6).
#
# Mirrors scripts/_run_olmo2_p24_eval_keep14_73.sh exactly (same harnesses, same
# tokenisation, same chat_template=False / --add_bos 0 protocol, same 8-GPU
# shard+merge convention), swapping arm/ckpt/output_name.
#
# Pre/post pair (single-variable, SAME node + SAME arch = H20 cc9.0 / zwfy6):
#   pre  : outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt
#          -> already scored on THIS disk as 7B_keep8_step121000_v2{,_know}
#             (PPL=13.332938, core6=0.5232843; per-item preds retained).
#             We do NOT re-run pre; we only ASSERT the anchor is complete.
#   post : outputs/olmo2_p24_sft_keep8fresh2/final.pt  (34.15 GB, step=842 keep=8 fresh=2)
#
# ⚠️ CAVEAT (carry into every report): keep8's pre-SFT anchor is step121000, NOT
# 200k -- keep8 never reached 200k (status/PAPERB_TABLE4_BUDGET_DEFECT.md). That
# is FINE here because this experiment holds the arm's OWN pre-SFT ckpt fixed and
# measures the SFT delta on it. It is NOT valid for compute-matched depth
# comparisons against keep14 (200k).
#
# Output name convention (matches paper Table 4 anchor naming):
#   PPL                  olmo2_ppl_results/<NAME>/summary.json
#   core6                olmo2_downstream_results/<NAME>/summary.json
#   know5                olmo2_downstream_results/<NAME>_know/summary.json
#   MMLU letter+content  olmo2_mmlu_content_results/<NAME>/summary.json
#   PopQA + TriviaQA     olmo2_closedbook_results/<NAME>/summary.json
# where NAME = "7B_p24_sft_keep8fresh2_final".
#
# Per-item predictions retained for pre/post pairing (McNemar, paired bootstrap):
#   downstream MC:  per_example_<task>.jsonl   (via --save_per_example)
#   MMLU content :  per_example_mmlu.jsonl     (default in harness)
#   closed-book  :  per_example_<task>.jsonl   (default in harness)
#
# Shard invariant: every eval must have 8/8 shard{i}of8.json before merge; a
# partial merge is a HARD ERROR (memory: kill-remote-gpu-job-by-pid-not-pkill).
#
# Launch:
#   setsid nohup bash scripts/_run_olmo2_p24_eval_sft_keep8_73.sh \
#     > logs/p24_eval_sft_keep8_73.log 2>&1 &
set -u

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
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

PRE_NAME="7B_keep8_step121000_v2"
POST_CKPT="outputs/olmo2_p24_sft_keep8fresh2/final.pt"
POST_NAME="7B_p24_sft_keep8fresh2_final"

# ---- pre-flight: post ckpt must exist ----
if [ ! -f "$POST_CKPT" ]; then
  echo "[$(date '+%F %T')] FATAL: post-SFT ckpt not found: $POST_CKPT"; exit 2
fi

# ---- pre-flight: the pre-SFT anchor must be COMPLETE (incl. per-item preds) ----
# If the anchor is incomplete we must NOT proceed silently: the whole point of
# this battery is the pre->post pairing, which needs per-item files on both sides.
ANCHOR_OK=1
for f in "olmo2_ppl_results/$PRE_NAME/summary.json" \
         "olmo2_downstream_results/$PRE_NAME/summary.json" \
         "olmo2_downstream_results/${PRE_NAME}_know/summary.json" \
         "olmo2_mmlu_content_results/$PRE_NAME/summary.json" \
         "olmo2_closedbook_results/$PRE_NAME/summary.json"; do
  if [ ! -f "$f" ]; then echo "[$(date '+%F %T')] ANCHOR MISSING: $f"; ANCHOR_OK=0; fi
done
for f in "olmo2_downstream_results/$PRE_NAME/per_example_hellaswag.jsonl" \
         "olmo2_mmlu_content_results/$PRE_NAME/per_example_mmlu.jsonl" \
         "olmo2_closedbook_results/$PRE_NAME/per_example_popqa.jsonl" \
         "olmo2_closedbook_results/$PRE_NAME/per_example_triviaqa.jsonl"; do
  if [ ! -f "$f" ]; then echo "[$(date '+%F %T')] ANCHOR PER-ITEM MISSING: $f"; ANCHOR_OK=0; fi
done
if [ $ANCHOR_OK -eq 0 ]; then
  echo "[$(date '+%F %T')] FATAL: pre-SFT anchor $PRE_NAME incomplete; pairing impossible. ABORT."
  exit 3
fi
echo "[$(date '+%F %T')] pre-SFT anchor $PRE_NAME verified complete (summaries + per-item preds)."

# assert_8shards <results_root> <NAME> <shard_pat>
#   Confirms all 8 shard files exist for a given eval; if not, ABORT the merge
#   (partial-merge = silent contamination). Called before every merge step.
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
  echo "[$(date '+%F %T')] SHARD ASSERT OK: $D 8/8"
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
  echo "[$(date '+%F %T')] downstream summary ($NAME):"; cat "$D/summary.json" 2>/dev/null | head -60; echo
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

echo "[$(date '+%F %T')] ======== P2.4 keep8fresh2 POST-SFT EVAL BATTERY (.73) START ========"
echo "[$(date '+%F %T')] pre anchor : $PRE_NAME (PPL=13.332938, core6=0.5232843)"
echo "[$(date '+%F %T')] post ckpt  : $POST_CKPT"
echo "[$(date '+%F %T')] PREDICTION : dPPL% ~= +14.0% => post-PPL ~= 15.20 (n=3 fit, r=0.998)"
echo "[$(date '+%F %T')] (reported as-observed; nothing is tuned to hit it)"

run_ppl        "$POST_NAME"       "$POST_CKPT"
run_downstream "$POST_NAME"       "$POST_CKPT" "$CORE"
run_downstream "${POST_NAME}_know" "$POST_CKPT" "$KNOW"
run_mmlu_content "$POST_NAME"     "$POST_CKPT"
run_closedbook "$POST_NAME"       "$POST_CKPT"

echo "[$(date '+%F %T')] ======== P2.4 keep8fresh2 POST-SFT EVAL BATTERY (.73) DONE ========"
echo "[$(date '+%F %T')] Outputs:"
echo "  post-SFT  PPL:            olmo2_ppl_results/$POST_NAME/summary.json"
echo "  post-SFT  core6+know5:    olmo2_downstream_results/${POST_NAME}{,_know}/summary.json"
echo "  post-SFT  MMLU dual:      olmo2_mmlu_content_results/$POST_NAME/summary.json"
echo "  post-SFT  closedbook:     olmo2_closedbook_results/$POST_NAME/summary.json"
