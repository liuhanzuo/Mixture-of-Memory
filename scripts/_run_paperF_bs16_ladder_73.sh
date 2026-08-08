#!/usr/bin/env bash
# Paper F -- bs16 ladder eval for flip-rate vs damage analysis.
# Runs bs16 for 5 rungs: base_full, keep14@200k, keep12@124k, keep10@83.5k, keep8@121k
# Node: .73 (28.85.35.73, 8xH20, zwfy6 disk)
# Protocol: identical to v2/bs8 runs except --batch_size 16 (ONLY change).
# Output dirs: 7B_{rung}_bs16   (NEVER overwrites existing dirs)
#
# Shard asserts: 8/8 shards mandatory before merge.
# n_scored asserts: hellaswag=10042 arc_challenge=1172 arc_easy=2376
#                   piqa=1838 openbookqa=500 winogrande=1267
set -u

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/OLMo-2-1124-7B
TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
BS=16

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

# Expected n_scored per task (for post-merge assertion)
declare -A EXPECTED_N
EXPECTED_N[hellaswag]=10042
EXPECTED_N[arc_challenge]=1172
EXPECTED_N[arc_easy]=2376
EXPECTED_N[piqa]=1838
EXPECTED_N[openbookqa]=500
EXPECTED_N[winogrande]=1267

assert_8shards () {
  local D="olmo2_downstream_results/$1"
  local MISS=0
  for g in 0 1 2 3 4 5 6 7; do
    [ -f "$D/shard${g}of8.json" ] || { echo "[SHARD MISSING] $D/shard${g}of8.json"; MISS=$((MISS+1)); }
  done
  [ $MISS -eq 0 ] && return 0
  echo "[ABORT] $MISS/8 shards missing for $1 -- NOT merging"
  return 1
}

assert_nscored () {
  local NAME="$1"
  local SUMMARY="olmo2_downstream_results/$NAME/summary.json"
  [ -f "$SUMMARY" ] || { echo "[ASSERT_N] summary.json not found for $NAME"; return 1; }
  $PY - <<PYEOF
import json, sys
with open("$SUMMARY") as f:
    s = json.load(f)
tasks = s.get("tasks", {})
expected = {"hellaswag":10042,"arc_challenge":1172,"arc_easy":2376,"piqa":1838,"openbookqa":500,"winogrande":1267}
fail = 0
for t, n in expected.items():
    got = tasks.get(t, {}).get("n_scored", -1)
    if got != n:
        print(f"[N_SCORED MISMATCH] {t}: expected {n}, got {got}")
        fail += 1
    else:
        print(f"[N_SCORED OK] {t}: {got}")
if fail > 0:
    print(f"[ASSERT_N FAILED] {fail} tasks with wrong n_scored for $NAME")
    sys.exit(1)
else:
    print(f"[ASSERT_N PASSED] all n_scored correct for $NAME")
PYEOF
}

# prepare datasets once
echo "[$(date '+%F %T')] prepare_data"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
  > logs/paperF_bs16_prepare.log 2>&1
tail -5 logs/paperF_bs16_prepare.log

# =====================================================================
# Config rows: "NAME|keep_front|n_fresh|ckpt"
# ckpt="" => base mode (no --ckpt, no --keep_front_layers/--n_fresh_layers)
# =====================================================================
CONFIGS=(
  "7B_base_full_bs16|||"
  "7B_keep14_step200000_bs16|14|2|outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt"
  "7B_keep12_step124000_bs16|12|2|outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt"
  "7B_keep10_step83500_bs16|10|2|outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt"
  "7B_keep8_step121000_bs16|8|2|outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt"
)

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME KFL NFL CKPT <<< "$row"
  echo "=========================================================="
  echo "[$(date '+%F %T')] CONFIG: $NAME  kfl=${KFL:-BASE}  nfl=${NFL:-BASE}  ckpt=${CKPT:-NONE}"

  # Safety: never overwrite existing results
  if [ -f "olmo2_downstream_results/$NAME/summary.json" ]; then
    echo "[$(date '+%F %T')] $NAME ALREADY DONE -- SKIPPING (not overwriting)"
    continue
  fi

  # Build arch args
  if [ -z "$CKPT" ]; then
    # base mode
    ARCH_ARGS=""
  else
    ARCH_ARGS="--ckpt $CKPT --keep_front_layers $KFL --n_fresh_layers $NFL"
  fi

  # Fan out 8 shards
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g \
    $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" $ARCH_ARGS \
      --tasks "$TASKS" \
      --num_shards 8 --shard_index $g \
      --batch_size $BS \
      --add_bos 0 \
      --save_per_example \
      --output_name "$NAME" \
      > "logs/paperF_bs16_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] shards done for $NAME; asserting 8/8..."
  assert_8shards "$NAME" || { echo "[FATAL] merge aborted for $NAME"; continue; }

  echo "[$(date '+%F %T')] merging $NAME..."
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1

  echo "[$(date '+%F %T')] asserting n_scored for $NAME..."
  assert_nscored "$NAME" || { echo "[FATAL] n_scored assertion failed for $NAME"; }

  echo "[$(date '+%F %T')] summary for $NAME:"
  cat "olmo2_downstream_results/$NAME/summary.json" 2>/dev/null
  echo
done

echo "[$(date '+%F %T')] ALL DONE -- paperF bs16 ladder"
echo "Dirs written:"
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME _ _ _ <<< "$row"
  [ -f "olmo2_downstream_results/$NAME/summary.json" ] && echo "  DONE: olmo2_downstream_results/$NAME" || echo "  MISSING: $NAME"
done
