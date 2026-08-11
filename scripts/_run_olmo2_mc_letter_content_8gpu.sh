#!/usr/bin/env bash
# ============================================================================
# paperG gate-2 (task #248) — non-MMLU letter-vs-content MC eval driver.
#
# Runs scripts/eval_olmo2_mc_letter_content.py over the SIX item-aligned OLMo-2-7B
# prune-then-heal arms (the same batch as the MMLU letter/content table and
# gate-3), 8 GPU shards per arm, all six MC tasks under one model load per shard.
#
# Output dir names deliberately MIRROR olmo2_mmlu_content_results/<name> so the
# paired MMLU-vs-here comparison is a name lookup, not a mapping table.
#
# Protocol: chat_template=False, add_bos=0, fp32 weights + bf16 autocast.
#
# env in:  ROOT  node project root (zwfy6 real path on .73/.82/.104)
#          PY    python (default conda torch-base; .venv is broken on H20)
#          BS    per-shard batch size (default 48)
#          TASKS comma-separated task list
#          ARMS  space-separated arm keys (default all six)
# ============================================================================
set -u
ROOT="${ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
BS="${BS:-48}"
NGPU="${NGPU:-8}"
N_BOOT="${N_BOOT:-10000}"
TASKS="${TASKS:-arc_challenge,arc_easy,openbookqa,commonsense_qa,piqa,winogrande}"
ARMS="${ARMS:-base keep8 keep10 keep12 keep14 shortgpt16}"
SCRIPT=scripts/eval_olmo2_mc_letter_content.py
cd "$ROOT" || exit 1

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs olmo2_mc_letter_content_results "$HF_DATASETS_CACHE"

# arm key -> "output_name|ckpt|keep_front|n_fresh"   (ckpt "-" = full base)
arm_spec() {
  case "$1" in
    base)       echo "7B_base|-|-|-" ;;
    keep8)      echo "7B_keep8_step121000|outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt|8|2" ;;
    keep10)     echo "7B_keep10_step83500|outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt|10|2" ;;
    keep12)     echo "7B_keep12_step124000|outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt|12|2" ;;
    keep14)     echo "7B_keep14_step200000|outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt|14|2" ;;
    shortgpt16) echo "7B_shortgpt16_step200000|outputs/olmo2_probe2_7B_shortgpt16/step200000.pt|16|0" ;;
    *) echo ""; return 1 ;;
  esac
}

echo "[$(date '+%F %T')] gate-2 letter/content driver: root=$ROOT bs=$BS tasks=$TASKS"
echo "[$(date '+%F %T')] arms: $ARMS"

# populate the dataset cache ONCE (CPU) so 8 shards do not race on the builder
$PY $SCRIPT --prepare_data --tasks "$TASKS" \
  > logs/gate2_mc_lc_prepare.log 2>&1 || { echo "PREPARE FAILED"; exit 1; }
tail -8 logs/gate2_mc_lc_prepare.log

for ARM in $ARMS; do
  SPEC="$(arm_spec "$ARM")" || { echo "unknown arm $ARM"; exit 1; }
  NAME="$(echo "$SPEC" | cut -d'|' -f1)"
  CKPT="$(echo "$SPEC" | cut -d'|' -f2)"
  KF="$(echo "$SPEC" | cut -d'|' -f3)"
  NF="$(echo "$SPEC" | cut -d'|' -f4)"

  CKARG=""
  if [ "$CKPT" != "-" ]; then
    if [ ! -f "$CKPT" ]; then
      echo "[$(date '+%F %T')] FATAL: ckpt missing on this disk: $ROOT/$CKPT"; exit 1
    fi
    CKARG="--ckpt $CKPT --keep_front_layers $KF --n_fresh_layers $NF"
  fi

  echo "[$(date '+%F %T')] ===== $ARM -> $NAME (ck='$CKPT') ====="
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $SCRIPT \
      --base_model "$BASE" $CKARG \
      --tasks "$TASKS" --num_shards $NGPU --shard_index $g \
      --batch_size $BS --output_name "$NAME" \
      > "logs/gate2_mc_lc_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  # merge asserts 8/8 shards present, n_scored == expected, n_nan == 0 per task,
  # and RAISES rather than merging a partial set.
  $PY $SCRIPT --merge --output_name "$NAME" --tasks "$TASKS" \
      --num_shards $NGPU --n_boot "$N_BOOT" \
      2>&1 | tee "logs/gate2_mc_lc_${NAME}_merge.log"
  if ! grep -q "\[merge\]" "logs/gate2_mc_lc_${NAME}_merge.log"; then
    echo "[$(date '+%F %T')] MERGE FAILED for $NAME"; exit 1
  fi
done

echo "[$(date '+%F %T')] ===== gate-2 letter/content ALL ARMS DONE ====="
