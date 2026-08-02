#!/usr/bin/env bash
# ============================================================================
# Paper B P0.6 — dual-protocol (letter + content-text) MMLU re-eval driver.
#
# For ONE arm (base OR a prune-then-heal ckpt), for each STEP in $STEPS, fan out
# 8 GPU shards of scripts/eval_olmo2_mmlu_content.py over cais/mmlu "all" (14,042
# items) and merge. The merge writes:
#   olmo2_mmlu_content_results/<TAG>_step<STEP>/summary.json  (letter_acc,
#     content_raw_acc, content_norm_acc, above-chance, within-arm McNemar +
#     paired-bootstrap CI, 57-subject breakdown)
#   olmo2_mmlu_content_results/<TAG>_step<STEP>/per_example_mmlu.jsonl (paired
#     per-item record, letter + content, stable item_ids -> cross-arm pairing).
#
# Base protocol: OLMo-2 is a BASE LM -> add_bos=0, chat_template=False, NO SFT.
# Same tokeniser / truncation / item_ids as scripts/eval_olmo2_probe2_downstream.py
# so letter numbers reproduce the published letter-protocol MMLU item-for-item.
#
# ── env knobs ────────────────────────────────────────────────────────────────
#   BASE           pretrained OLMo-2 path (default ../models/OLMo-2-1124-7B)
#   CKDIR          ckpt dir; leave EMPTY for full-base mode (no --ckpt)
#   STEPS          space-sep steps (ignored in base mode). 200000 -> final.pt.
#   KEEP_FRONT     pruned-shell kept front layers (keep8->8 keep14->14 SGPT->16)
#   N_FRESH        pruned-shell fresh layers   (keepN->2, ShortGPT->0)
#   KEEP_INDICES   ShortGPT selected indices (informational meta only)
#   TAG            output-name prefix (e.g. 7B_keep14, 7B_base, 7B_shortgpt16)
#   CONTENT_DESC   content prompt: full (default) | none
#   BS             per-shard batch size (default 16)
#   NGPU           #GPUs to shard over (default 8)
#   PY             python (default $WD/.venv/bin/python)
# ============================================================================
set -u
WD="${WD:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$WD" || exit 1
PY="${PY:-$WD/.venv/bin/python}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
CKDIR="${CKDIR:-}"                 # empty => full-base mode
STEPS="${STEPS:-200000}"
KEEP_FRONT="${KEEP_FRONT:-}"
N_FRESH="${N_FRESH:-}"
KEEP_INDICES="${KEEP_INDICES:-}"
TAG="${TAG:-7B_base}"
CONTENT_DESC="${CONTENT_DESC:-full}"
BS="${BS:-16}"
NGPU="${NGPU:-8}"
N_BOOT="${N_BOOT:-10000}"
SCRIPT=scripts/eval_olmo2_mmlu_content.py

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=$WD/data/hf_datasets_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs olmo2_mmlu_content_results "$HF_DATASETS_CACHE"

# extra CLI for pruned shell shape (empty in base mode)
SHELL_ARGS=""
[ -n "$KEEP_FRONT" ] && SHELL_ARGS="$SHELL_ARGS --keep_front_layers $KEEP_FRONT"
[ -n "$N_FRESH" ]    && SHELL_ARGS="$SHELL_ARGS --n_fresh_layers $N_FRESH"
[ -n "$KEEP_INDICES" ] && SHELL_ARGS="$SHELL_ARGS --keep_indices $KEEP_INDICES"

resolve_ckpt() {  # echo ckpt path for a step, or empty
  local s="$1"
  if [ -f "$CKDIR/step${s}.pt" ]; then echo "$CKDIR/step${s}.pt";
  elif [ "$s" = "200000" ] && [ -f "$CKDIR/final.pt" ]; then echo "$CKDIR/final.pt";
  else echo ""; fi
}

run_one() {  # $1=NAME  $2=ckpt-arg-string ("" for base)
  local NAME="$1"; local CKARG="$2"
  echo "[$(date '+%F %T')] ===== $NAME (ckarg='$CKARG') ====="
  # prepare cais/mmlu cache ONCE (CPU, proxy) to avoid an 8-way download race
  $PY $SCRIPT --prepare_data --content_desc "$CONTENT_DESC" \
      > "logs/olmo2_mmlu_content_${NAME}_prepare.log" 2>&1
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $SCRIPT \
      --base_model "$BASE" $CKARG $SHELL_ARGS \
      --content_desc "$CONTENT_DESC" \
      --num_shards $NGPU --shard_index $g --batch_size $BS \
      --output_name "$NAME" \
      > "logs/olmo2_mmlu_content_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY $SCRIPT --merge --output_name "$NAME" --n_boot "$N_BOOT" 2>&1
  echo "[$(date '+%F %T')] $NAME summary:";
  cat "olmo2_mmlu_content_results/${NAME}/summary.json" 2>/dev/null | head -c 900; echo
}

if [ -z "$CKDIR" ]; then
  # ---- full-base mode (no ckpt) ----
  run_one "${TAG}" ""
else
  for STEP in $STEPS; do
    CKPT="$(resolve_ckpt "$STEP")"
    if [ -z "$CKPT" ]; then
      echo "[$(date '+%F %T')] WARNING: no ckpt for step $STEP in $CKDIR -> SKIP"; continue
    fi
    run_one "${TAG}_step${STEP}" "--ckpt $CKPT"
  done
fi
echo "[$(date '+%F %T')] ===== P0.6 content-MMLU eval ALL DONE (TAG=$TAG) ====="
