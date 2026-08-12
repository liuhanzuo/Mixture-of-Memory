#!/usr/bin/env bash
# ============================================================================
# Paper B task #181 — keep14+fresh2 SEED VARIANCE (seed42 vs seed1234) @ step200000
#
# WHY THIS DRIVER RE-RUNS SEED 42 TOO
# -----------------------------------
# The whole value of the seed1234 run is a SAME-PROTOCOL delta against the
# existing seed42 keep14@200000 arm. The archived seed42 numbers were produced
# on 2026-07-28 on node .252 (retired) with $WD/.venv/bin/python, and LOCAL's
# .venv no longer has torch (2026-08-04 fact, CLAUDE.md). Running seed1234 on
# LOCAL/conda while comparing to a .252/.venv archive would measure
# (seed + node + toolchain) jointly.
#
# status/PAPERB_WITHIN_DISK_FLOOR_V3.md established that a SINGLE driver
# revision on the same disk+arch is bit-deterministic (0 flips), and the only
# non-zero within-disk comparison crossed a DRIVER boundary. So the clean design
# is: run BOTH seeds here, back to back, same GPUs, same python, same driver,
# same commit. Then the delta is attributable to the seed alone.
#
# The re-run of seed42 is ALSO a provenance check: if it reproduces the archived
# .252 numbers exactly, the archive is portable; if it does not, we have
# quantified the node/toolchain term and we still have a valid seed contrast
# (because both arms in the contrast were made HERE).
#
# AXES (only those with a real seed42 baseline on this disk)
#   (1) held-out NTP PPL      eval_olmo2_probe2_ppl.py         dolmino_now_val
#   (2) core6 downstream MC   eval_olmo2_probe2_downstream.py  --save_per_example
#   (3) know5 downstream MC   eval_olmo2_probe2_downstream.py  --save_per_example
#   (4) MMLU letter+content   eval_olmo2_mmlu_content.py       14042 items
#   (5) OOD PPL wikitext103   eval_olmo2_probe2_ppl.py
#   (6) OOD PPL pg19          eval_olmo2_probe2_ppl.py
# NOT run: closed-book PopQA/TriviaQA (seed42 baseline exists only on zwfy6 and
# the HF nq_open/popqa generation path is not cached on wzc1) and the paperC
# mc_letter_content axis (that is a paperC gate, not a Paper B seed axis).
#
# PROTOCOL (identical to the archived seed42 rows)
#   chat_template=False (OLMo-2 is a BASE LM, no SFT), --add_bos 0,
#   fp32 weights + bf16 autocast, 8 shards [i::8], batch_size 4 (ppl) / 8
#   (downstream) / 16 (mmlu), merge asserts 8/8 shards.
#
# GPU budget: LOCAL 8xL20A ONLY. Never touches .21/.73/.82/.104.
# ============================================================================
set -u
WD=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$WD" || exit 1

# LOCAL .venv has no torch since 2026-08-04 -> conda (torch 2.13.0, sm_100 ok).
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
VAL=data/dolmino_now_val.npy
WT=data/ood_ppl/wikitext103_test.npy
PG=data/ood_ppl/pg19_test.npy
NGPU=8
N_BOOT=10000

export HF_DATASETS_CACHE=$WD/data/hf_datasets_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs olmo2_ppl_results olmo2_downstream_results olmo2_mmlu_content_results \
         ood_ppl_results "$HF_DATASETS_CACHE"

CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW_TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

# arm rows: "TAG|CKPT"
ARMS=(
  "s42|outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt"
  "s1234|outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step200000.pt"
)
SUFFIX="${SUFFIX:-_sv181}"   # namespace so nothing archived is ever clobbered

log(){ echo "[$(date '+%F %T')] $*"; }

# refuse to clobber an existing result dir (loud, not silent)
guard(){ # $1=root $2=name
  if [ -d "$1/$2" ]; then
    log "FATAL: $1/$2 already exists (would clobber). Change SUFFIX."; exit 2
  fi
}

# assert 8/8 shard files landed before any merge
assert_shards(){ # $1=dir
  local n; n=$(ls "$1"/shard*of8.json 2>/dev/null | wc -l)
  if [ "$n" -ne 8 ]; then
    log "FATAL: only $n/8 shards in $1 -- refusing to merge a partial set"; exit 4
  fi
  log "OK 8/8 shards in $1"
}

for row in "${ARMS[@]}"; do
  TAG="${row%%|*}"; CKPT="${row#*|}"
  [ -f "$CKPT" ] || { log "FATAL: ckpt missing: $CKPT"; exit 3; }
  log "################ ARM $TAG  ckpt=$CKPT ################"

  # ---------------- (1) in-domain held-out PPL ----------------
  NAME="keep14_${TAG}_step200000${SUFFIX}"
  guard olmo2_ppl_results "$NAME"
  log "(1) in-domain PPL -> $NAME"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --keep_front_layers 14 --n_fresh_layers 2 \
      --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
      --output_name "$NAME" \
      > "logs/sv181_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_shards "olmo2_ppl_results/$NAME"
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1

  # ---------------- (2) core6 downstream ----------------
  NAME="keep14_${TAG}_step200000${SUFFIX}"
  guard olmo2_downstream_results "$NAME"
  log "(2) core6 downstream -> $NAME"
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS" \
      > "logs/sv181_prep_core.log" 2>&1
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE_TASKS" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --save_per_example --output_name "$NAME" \
      > "logs/sv181_core_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_shards "olmo2_downstream_results/$NAME"
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1

  # ---------------- (3) know5 downstream ----------------
  NAMEK="keep14_${TAG}_step200000${SUFFIX}_know"
  guard olmo2_downstream_results "$NAMEK"
  log "(3) know5 downstream -> $NAMEK"
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$KNOW_TASKS" \
      > "logs/sv181_prep_know.log" 2>&1
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$KNOW_TASKS" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --save_per_example --output_name "$NAMEK" \
      > "logs/sv181_know_${NAMEK}_shard${g}.log" 2>&1 &
  done
  wait
  assert_shards "olmo2_downstream_results/$NAMEK"
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAMEK" 2>&1

  # ---------------- (4) MMLU letter + content dual ----------------
  NAMEM="keep14_${TAG}_step200000${SUFFIX}"
  guard olmo2_mmlu_content_results "$NAMEM"
  log "(4) MMLU letter+content -> $NAMEM"
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data --content_desc full \
      > "logs/sv181_prep_mmlu.log" 2>&1
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --keep_front_layers 14 --n_fresh_layers 2 \
      --content_desc full --num_shards 8 --shard_index $g --batch_size 16 \
      --output_name "$NAMEM" \
      > "logs/sv181_mmlu_${NAMEM}_shard${g}.log" 2>&1 &
  done
  wait
  assert_shards "olmo2_mmlu_content_results/$NAMEM"
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAMEM" \
      --n_boot "$N_BOOT" 2>&1

  # ---------------- (5)(6) OOD PPL ----------------
  for c in "wikitext103:$WT" "pg19:$PG"; do
    cn="${c%%:*}"; cp="${c##*:}"
    NAMEO="keep14_${TAG}_step200000${SUFFIX}_${cn}"
    guard ood_ppl_results "$NAMEO"
    log "(ood) $cn -> $NAMEO"
    # archived seed42 OOD rows were num_shards=1 batch_size=8 -> keep verbatim
    CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --keep_front_layers 14 --n_fresh_layers 2 \
      --val_path "$cp" --num_shards 1 --shard_index 0 --batch_size 8 \
      --output_name "$NAMEO" --results_root ood_ppl_results \
      > "logs/sv181_ood_${NAMEO}.log" 2>&1
    $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAMEO" \
        --results_root ood_ppl_results 2>&1
  done

  log "################ ARM $TAG DONE ################"
done

log "===== task #181 seed-variance battery ALL DONE ====="
