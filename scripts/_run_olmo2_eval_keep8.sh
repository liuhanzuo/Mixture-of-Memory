#!/usr/bin/env bash
# FULL base-protocol eval for the keep8+fresh2 prune-then-heal arm (Paper B depth
# ladder, shallowest point = 10-layer shell). Identical harness/python to the
# keep14@200k / ShortGPT eval drivers -- only the ckpt dir, the pruned-shell shape
# (keep_front_layers=8 n_fresh_layers=2 -> 10L) and the output NAME differ.
#
# For each requested step it runs, 8-GPU sharded and sequential:
#   (1) held-out NTP PPL   (2) core 6-task downstream   (3) knowledge 5-task incl MMLU
#
# Default STEPS eval a 3-point recent ladder (100k/110k/121k) so the knowledge-axis
# plateau can be documented (flat within noise over ~21k steps), mirroring how
# keep10 (83.5k) / keep12 (124k) plateau stops were established. Highest step =
# the headline keep8 endpoint. Missing ckpts are skipped with a warning.
set -u
WD="${WD:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$WD" || exit 1
PY="${PY:-$WD/.venv/bin/python}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
CKDIR="${CKDIR:-outputs/olmo2_probe2_7B_keep8fresh2}"
VAL="${VAL:-data/dolmino_now_val.npy}"
STEPS="${STEPS:-121000 110000 100000}"
# keep8 shell = 8 kept front layers + 2 fresh = 10-layer shell for build_pruned_shell.
KEEP_FRONT="${KEEP_FRONT:-8}"
N_FRESH="${N_FRESH:-2}"
TAG="${TAG:-7B_keep8}"

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=$WD/data/hf_datasets_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs olmo2_ppl_results olmo2_downstream_results "$HF_DATASETS_CACHE"

CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW_TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

resolve_ckpt() {  # echo the ckpt path for a step, or empty if absent
  local s="$1"
  if [ -f "$CKDIR/step${s}.pt" ]; then echo "$CKDIR/step${s}.pt";
  elif [ "$s" = "200000" ] && [ -f "$CKDIR/final.pt" ]; then echo "$CKDIR/final.pt";
  else echo ""; fi
}

echo "[$(date '+%F %T')] ===== keep8 FULL eval start (ckdir=$CKDIR steps='$STEPS' keep_front=$KEEP_FRONT n_fresh=$N_FRESH) ====="

for STEP in $STEPS; do
  CKPT="$(resolve_ckpt "$STEP")"
  if [ -z "$CKPT" ]; then
    echo "[$(date '+%F %T')] WARNING: no ckpt for step $STEP in $CKDIR -> SKIP"
    continue
  fi
  NAME="${TAG}_step${STEP}"
  NAMEK="${NAME}_know"
  echo "[$(date '+%F %T')] ===== step $STEP  ckpt=$CKPT ====="

  # ---------- (1) held-out PPL ----------
  echo "[$(date '+%F %T')] (1) PPL $NAME"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --keep_front_layers "$KEEP_FRONT" --n_fresh_layers "$N_FRESH" \
      --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
      --output_name "$NAME" \
      > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] PPL done:"; cat "olmo2_ppl_results/${NAME}/summary.json" 2>/dev/null

  # ---------- (2) core 6-task ----------
  echo "[$(date '+%F %T')] (2) core downstream $NAME"
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS" \
    > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE_TASKS" \
      --keep_front_layers "$KEEP_FRONT" --n_fresh_layers "$N_FRESH" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] core done:"; cat "olmo2_downstream_results/${NAME}/summary.json" 2>/dev/null | head -c 400; echo

  # ---------- (3) knowledge 5-task (incl MMLU) ----------
  echo "[$(date '+%F %T')] (3) knowledge $NAMEK"
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$KNOW_TASKS" \
    > "logs/olmo2_downstream_${NAMEK}_prepare.log" 2>&1
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$KNOW_TASKS" \
      --keep_front_layers "$KEEP_FRONT" --n_fresh_layers "$N_FRESH" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAMEK" \
      > "logs/olmo2_downstream_${NAMEK}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAMEK" 2>&1
  echo "[$(date '+%F %T')] knowledge done:"; cat "olmo2_downstream_results/${NAMEK}/summary.json" 2>/dev/null | head -c 400; echo
done

echo "[$(date '+%F %T')] ===== keep8 FULL eval ALL DONE ====="
