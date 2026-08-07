#!/usr/bin/env bash
# Paper B P2.4 / Task #189 — Table 4 ladder pre-SFT re-eval on .73 (zwfy6/H20)
#   Purpose: extend the H20 vs L20A "flip-count scales with pruning damage"
#     observation from n=2 (full32 base = 10 flips, keep14 = 28 flips) to n=5
#     by producing H20 per-item preds for keep8 / keep10 / keep12 / shortgpt16.
#   Byte-identical mirror of `scripts/_run_olmo2_p24_eval_keep14_73.sh` — same
#   harnesses, tokenisation, chat_template=False, add_bos=false, bf16 autocast,
#   8-GPU shard+merge, --save_per_example, assert_8shards invariant.
#   Only the arm loop changes: four rungs, pre-SFT only, `_v2` output names so
#   we NEVER overwrite the paper's Table 4 source summaries.
#
# Headline ckpts (per paperB/P0_7_AGGREGATE_AUDIT.md §2 — the P0 audit tags
# these as the paper Table 4 rungs; keep8/10/12 never reached step200000):
#   keep8       outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt      -> 7B_keep8_step121000_v2
#   keep10      outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt      -> 7B_keep10_step83500_v2
#   keep12      outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt     -> 7B_keep12_step124000_v2
#   shortgpt16  outputs/olmo2_probe2_7B_shortgpt16/step200000.pt       -> 7B_shortgpt16_step200000_v2
# (task text specified `step200000.pt` for all four rungs, but that file only
# exists for shortgpt16; the audit-canonical headline steps are used instead —
# see status/PAPERB_P24_LADDER_PREV2_EVAL.md §"path correction".)
#
# Output-name convention identical to the keep14 sibling (which writes without
# `_v2`); we append `_v2` so pre-existing summary.json files at the base names
# stay the record-of-truth for Table 4:
#   PPL              olmo2_ppl_results/<NAME>_v2/summary.json
#   core6            olmo2_downstream_results/<NAME>_v2/summary.json
#   know5            olmo2_downstream_results/<NAME>_v2_know/summary.json
#   MMLU dual        olmo2_mmlu_content_results/<NAME>_v2/summary.json
#   closedbook       olmo2_closedbook_results/<NAME>_v2/summary.json
#
# Per-item predictions retained (downstream --save_per_example; MMLU-content and
# closedbook harnesses write per-item files by default). Load-bearing for the
# task #189 cross-arch McNemar + paired bootstrap across the ladder.
#
# Launch (from .73, cd to zwfy6 root):
#   setsid nohup bash scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh \
#     > logs/p24_eval_ladder_prev2_73.log 2>&1 &
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

# Four rungs, headline steps per paperB/P0_7_AGGREGATE_AUDIT.md §2.
# Arrays are index-aligned: ARM_NAMES[i] labels, CKPTS[i] file paths, NAMES[i]
# output-name bases (the `_v2` suffix is appended per-harness helper below).
ARM_NAMES=(keep8 keep10 keep12 shortgpt16)
CKPTS=(
  "outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt"
  "outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt"
  "outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt"
  "outputs/olmo2_probe2_7B_shortgpt16/step200000.pt"
)
NAMES=(
  "7B_keep8_step121000_v2"
  "7B_keep10_step83500_v2"
  "7B_keep12_step124000_v2"
  "7B_shortgpt16_step200000_v2"
)

# ---- pre-flight ----
for CK in "${CKPTS[@]}"; do
  if [ ! -f "$CK" ]; then
    echo "[$(date '+%F %T')] FATAL: ckpt not found: $CK"; exit 2
  fi
done
echo "[$(date '+%F %T')] pre-flight OK: 4/4 ckpts present"

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
  return 0
}

# ---------------- eval helpers (8-GPU shard+merge; --save_per_example on MC) ----

# run_ppl NAME CKPT
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

# run_downstream NAME CKPT TASKS
# Uses --save_per_example so downstream per-task McNemar + paired bootstrap can be run.
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

# run_mmlu_content NAME CKPT  (both letter + content protocols; per-item default)
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

# run_closedbook NAME CKPT
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

echo "[$(date '+%F %T')] ======== P2.4 LADDER PRE-SFT _v2 EVAL BATTERY (.73) START ========"
echo "[$(date '+%F %T')] Rungs: ${ARM_NAMES[*]}"
echo "[$(date '+%F %T')] Ckpts: ${CKPTS[*]}"
echo "[$(date '+%F %T')] Names: ${NAMES[*]}"

# Serial across arms; parallel across 8 shards within each battery (~15-25 min/harness on H20).
# 4 arms × 5 harnesses = 20 harness invocations ≈ 4-6 h total.
for i in 0 1 2 3; do
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

echo "[$(date '+%F %T')] ======== P2.4 LADDER PRE-SFT _v2 EVAL BATTERY (.73) DONE ========"
echo "[$(date '+%F %T')] Outputs:"
for i in 0 1 2 3; do
  NAME="${NAMES[$i]}"
  echo "  ${ARM_NAMES[$i]}:"
  echo "    PPL:            olmo2_ppl_results/$NAME/summary.json"
  echo "    core6+know5:    olmo2_downstream_results/${NAME}{,_know}/summary.json"
  echo "    MMLU dual:      olmo2_mmlu_content_results/$NAME/summary.json"
  echo "    closedbook:     olmo2_closedbook_results/$NAME/summary.json"
done
