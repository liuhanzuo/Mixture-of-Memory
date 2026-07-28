#!/usr/bin/env bash
# FULL base-protocol eval for the ShortGPT prune-then-heal arm (Paper B external
# baseline). Same harness/scripts as the keep14@200k eval driver -- only the ckpt
# dir + the pruned-shell shape (keep_front_layers=16 n_fresh_layers=0) differ.
#
# For each requested step it runs, 8-GPU sharded and sequential:
#   (1) held-out NTP PPL   (2) core 6-task downstream   (3) knowledge 5-task incl MMLU
#
# Steps evaluated (mirrors the keep14 evaluated frontier):
#   step0      = pruned-but-NOT-healed (heal-free damage point; saved by the trainer
#                immediately after transplant)
#   128000 / 153500 / 200000 = keep14-matched heal checkpoints (the trainer
#                force-saves 128000/153500 via --extra_save_steps; 200000 = final)
# Override with STEPS="0 200000" etc. Missing ckpts are skipped with a warning
# (200000 falls back to final.pt, matching the keep14 driver).
set -u
WD="${WD:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$WD" || exit 1
PY="${PY:-$WD/.venv/bin/python}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
CKDIR="${CKDIR:-outputs/olmo2_probe2_7B_shortgpt16}"
VAL="${VAL:-data/dolmino_now_val.npy}"
STEPS="${STEPS:-0 128000 153500 200000}"
# ShortGPT shell = 16 kept layers + 0 fresh = 16-layer shell for build_pruned_shell.
KEEP_FRONT="${KEEP_FRONT:-16}"
N_FRESH="${N_FRESH:-0}"

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

echo "[$(date '+%F %T')] ===== ShortGPT FULL eval start (ckdir=$CKDIR steps='$STEPS') ====="

for STEP in $STEPS; do
  CKPT="$(resolve_ckpt "$STEP")"
  if [ -z "$CKPT" ]; then
    echo "[$(date '+%F %T')] WARNING: no ckpt for step $STEP in $CKDIR -> SKIP"
    continue
  fi
  NAME="7B_shortgpt16_step${STEP}"
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

echo "[$(date '+%F %T')] ===== ShortGPT FULL eval ALL DONE ====="
