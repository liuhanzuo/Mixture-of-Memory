#!/usr/bin/env bash
# Paper B P0 eval battery for .73 (diskB). Mirrors the harness of
# scripts/_run_distill_step5000_eval_then_restart.sh (same ROOT/PY/BASE, same
# proxy + HF_DATASETS_CACHE, same 8-GPU shard+merge for PPL and downstream MC),
# and ends by RESUMING the distill heal exactly like that script does.
#
# Phases (each eval phase ends with `|| true` so a single crash cannot block the
# Phase-4 distill resume):
#   Phase 0  build step0.pt (initial keep14+fresh2 student, identical to training
#            construction), single-process on GPU 0. rotation disabled -> never
#            deletes an existing trained ckpt in that output_dir.
#   Phase 1  eval step0.pt: held-out PPL (BS4, 8 shard+merge) + core6 + know5.
#            NAME=7B_keep14_step0 (know: 7B_keep14_step0_know).
#   Phase 2  mmlu_pro for base / keep14_step153500 / keep8_step44000 / step0
#            (each guarded by [ -f ]; base always runs). NAME=7B_<arm>_<step>_mmlupro.
#   Phase 3  keep8 MMLU trajectory backfill: keep8 step5000/15000/35000 (guarded)
#            -> core6+know5. NAME=7B_keep8_step<N> (know: _know). PPL already exists
#            for these; MMLU is added for the NLL-MMLU heal trajectory.
#   Phase 4  resume the distill heal from step5000.pt (BS=4 GA=4), verbatim from
#            _run_distill_step5000_eval_then_restart.sh.
#
# ⚠️ The distill training must ALREADY be KILLED and the 8 GPUs free before you
#    launch this (Phase 1-3 grab all 8 GPUs, Phase 4 hands them back to distill).
# Launch:
#   setsid nohup bash scripts/_run_olmo2_p0_evalbattery_73.sh >logs/p0_evalbattery_73.log 2>&1 &
set -u
ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE="../models/OLMo-2-1124-7B"
VAL=data/dolmino_now_val.npy

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs olmo2_ppl_results olmo2_downstream_results "$HF_DATASETS_CACHE"

CORE="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

# ---------------------------------------------------------------------------
# helpers (8-GPU shard + merge; identical command shape to the mirror harness)
# ---------------------------------------------------------------------------
# run_ppl NAME CKPT KEEP FRESH  -- held-out NTP PPL over 2048-tok windows
run_ppl () {
  local NAME=$1 CKPT=$2 KEEP=$3 FRESH=$4
  echo "[$(date '+%F %T')] --- PPL $NAME (ckpt=$CKPT keep=$KEEP fresh=$FRESH) ---"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --keep_front_layers "$KEEP" --n_fresh_layers "$FRESH" \
      --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
      --output_name "$NAME" \
      > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1 || true
  echo "[$(date '+%F %T')] PPL summary ($NAME):"; cat "olmo2_ppl_results/${NAME}/summary.json" 2>/dev/null; echo
}

# run_downstream NAME CKPT TASKS  -- likelihood MC (CKPT="" -> full-depth base mode)
run_downstream () {
  local NAME=$1 CKPT=$2 TASKS=$3
  echo "[$(date '+%F %T')] --- downstream $NAME (ckpt='${CKPT}' tasks=$TASKS) ---"
  local CKARG=()
  [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
    > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1 || true
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" "${CKARG[@]}" --tasks "$TASKS" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1 || true
  echo "[$(date '+%F %T')] downstream summary ($NAME):"; cat "olmo2_downstream_results/${NAME}/summary.json" 2>/dev/null; echo
}

echo "[$(date '+%F %T')] ============ P0 EVAL BATTERY START (.73) ============"

# ---------------------------------------------------------------------------
# Phase 0: build step0.pt (initial keep14+fresh2 student = training's step-0 state)
# ---------------------------------------------------------------------------
STEP0_DIR="outputs/olmo2_probe2_7B_keep14fresh2"
STEP0="$STEP0_DIR/step0.pt"
echo "[$(date '+%F %T')] --- PHASE 0: build $STEP0 ---"
if [ ! -f "$STEP0" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_olmo2_arch_probe2.py \
    --save_step0_and_exit \
    --data_path "$VAL" \
    --output_dir "$STEP0_DIR" \
    --model_path "$BASE" \
    --keep_front_layers 14 --n_fresh_layers 2 \
    > "logs/olmo2_build_step0.log" 2>&1 || true
fi
[ -f "$STEP0" ] && echo "[$(date '+%F %T')] step0.pt present: $STEP0" \
               || echo "[$(date '+%F %T')] WARNING: step0.pt NOT built (see logs/olmo2_build_step0.log)"

# ---------------------------------------------------------------------------
# Phase 1: eval step0.pt -- PPL + core6 + know5
# ---------------------------------------------------------------------------
echo "[$(date '+%F %T')] --- PHASE 1: eval step0.pt ---"
if [ -f "$STEP0" ]; then
  run_ppl        "7B_keep14_step0"      "$STEP0" 14 2 || true
  run_downstream "7B_keep14_step0"      "$STEP0" "$CORE" || true
  run_downstream "7B_keep14_step0_know" "$STEP0" "$KNOW" || true
else
  echo "[$(date '+%F %T')] Phase 1 SKIPPED (no step0.pt)"
fi

# ---------------------------------------------------------------------------
# Phase 2: mmlu_pro over base + keep14_step153500 + keep8_step44000 + step0
# ---------------------------------------------------------------------------
echo "[$(date '+%F %T')] --- PHASE 2: mmlu_pro ---"
# base (full-depth 32L; no --ckpt) always runs
run_downstream "7B_base_mmlupro" "" "mmlu_pro" || true

KEEP14_S153500="outputs/olmo2_probe2_7B_keep14fresh2/keep14_step153500.pt"
[ -f "$KEEP14_S153500" ] \
  && { run_downstream "7B_keep14_step153500_mmlupro" "$KEEP14_S153500" "mmlu_pro" || true; } \
  || echo "[$(date '+%F %T')] Phase 2 skip: $KEEP14_S153500 absent"

KEEP8_S44000="outputs/olmo2_probe2_7B_keep8fresh2/step44000.pt"
[ -f "$KEEP8_S44000" ] \
  && { run_downstream "7B_keep8_step44000_mmlupro" "$KEEP8_S44000" "mmlu_pro" || true; } \
  || echo "[$(date '+%F %T')] Phase 2 skip: $KEEP8_S44000 absent"

[ -f "$STEP0" ] \
  && { run_downstream "7B_keep14_step0_mmlupro" "$STEP0" "mmlu_pro" || true; } \
  || echo "[$(date '+%F %T')] Phase 2 skip: $STEP0 absent"

# ---------------------------------------------------------------------------
# Phase 3: keep8 MMLU trajectory backfill (core6 + know5 at step5000/15000/35000)
# ---------------------------------------------------------------------------
echo "[$(date '+%F %T')] --- PHASE 3: keep8 MMLU trajectory ---"
for N in 5000 15000 35000; do
  CK="outputs/olmo2_probe2_7B_keep8fresh2/step${N}.pt"
  if [ -f "$CK" ]; then
    run_downstream "7B_keep8_step${N}"      "$CK" "$CORE" || true
    run_downstream "7B_keep8_step${N}_know" "$CK" "$KNOW" || true
  else
    echo "[$(date '+%F %T')] Phase 3 skip: $CK absent"
  fi
done

echo "[$(date '+%F %T')] ============ EVAL BATTERY DONE — restarting distill from step5000.pt (BS=4 GA=4) ============"

# ---------------------------------------------------------------------------
# Phase 4: resume the distill heal from step5000.pt (verbatim from the mirror)
# ---------------------------------------------------------------------------
RESUME_FROM="$ROOT/outputs/olmo2_probe2_7B_keep14fresh2_distill/step5000.pt" \
BS=4 GA=4 RUN=1 \
PROJECT_ROOT="$ROOT" \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
MODEL_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
TEACHER_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
DATA_PATH=/dev/shm/dolmino_now15b.npy \
  bash scripts/_run_olmo2_keep14_distill_heal.sh
echo "[$(date '+%F %T')] ============ P0 EVAL BATTERY ORCH DONE (distill relaunched) ============"
