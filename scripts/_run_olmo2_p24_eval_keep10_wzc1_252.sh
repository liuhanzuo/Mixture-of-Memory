#!/usr/bin/env bash
# Paper B P2.4 / Task #189 (extension) — keep10 pre-SFT wzc1/L20A eval on .252
#   Purpose: complete the 5x2 damage-scaling grid. Cross-architecture audit
#     currently has n=4 wzc1-side pairs (full-32L / keep8 / keep14 / shortgpt16);
#     keep10 is the only rung still without a wzc1 pre-SFT eval because its
#     step83500.pt lived only on zwfy6. MAIN scp -O'd it 2026-08-08.
#   Byte-identical mirror of `_run_olmo2_p24_eval_ladder_wzc1_252.sh` — same
#     harnesses, tokenisation, chat_template=False, add_bos=false, bf16
#     autocast, 8-GPU shard+merge, --save_per_example, assert_8shards invariant.
#   Only the arm list swaps: single arm `keep10` at Table 4 headline
#     step83500 (matches the zwfy6 `_v2` sibling exactly, so the L20A vs H20
#     delta is arch-only, not arch+step).
#
# Load-bearing checkpoint provenance:
#   keep10      outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt   -> 7B_keep10_step83500_wzc1
#               (~36 GB; MATCHES the Table 4 headline step for keep10.
#                Transferred cross-disk from zwfy6:.73 via scp -O 2026-08-08.
#                md5 verified on both sides before eval start.)
#
# Output-name convention (all with `_wzc1` suffix on the pre-SFT anchor):
#   PPL              olmo2_ppl_results/<NAME>_wzc1/summary.json
#   core6            olmo2_downstream_results/<NAME>_wzc1/summary.json
#   know5            olmo2_downstream_results/<NAME>_wzc1_know/summary.json
#   MMLU dual        olmo2_mmlu_content_results/<NAME>_wzc1/summary.json
#   closedbook       olmo2_closedbook_results/<NAME>_wzc1/summary.json
#
# Launch (on .252, cd to wzc1 root):
#   setsid nohup bash scripts/_run_olmo2_p24_eval_keep10_wzc1_252.sh \
#     > logs/p24_eval_keep10_wzc1_252.log 2>&1 &
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

# Single rung.
ARM_NAMES=(keep10)
CKPTS=(
  "outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt"
)
NAMES=(
  "7B_keep10_step83500_wzc1"
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
echo "[$(date '+%F %T')] pre-flight OK: 1/1 ckpt present; BASE and VAL present"

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

echo "[$(date '+%F %T')] ======== P2.4 KEEP10 PRE-SFT _wzc1 EVAL BATTERY (.252, L20A) START ========"
echo "[$(date '+%F %T')] Rung: ${ARM_NAMES[*]}"
echo "[$(date '+%F %T')] Ckpt: ${CKPTS[*]}"
echo "[$(date '+%F %T')] Name: ${NAMES[*]}"
echo "[$(date '+%F %T')] NOTE: step83500 matches the Table 4 headline / zwfy6 _v2 sibling exactly; cross-arch delta is arch-only."

for i in 0; do
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

echo "[$(date '+%F %T')] ======== P2.4 KEEP10 PRE-SFT _wzc1 EVAL BATTERY (.252) DONE ========"
echo "[$(date '+%F %T')] Outputs:"
for i in 0; do
  NAME="${NAMES[$i]}"
  echo "  ${ARM_NAMES[$i]}:"
  echo "    PPL:            olmo2_ppl_results/$NAME/summary.json"
  echo "    core6+know5:    olmo2_downstream_results/${NAME}{,_know}/summary.json"
  echo "    MMLU dual:      olmo2_mmlu_content_results/$NAME/summary.json"
  echo "    closedbook:     olmo2_closedbook_results/$NAME/summary.json"
done
