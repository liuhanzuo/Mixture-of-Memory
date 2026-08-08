#!/usr/bin/env bash
# Paper B — keep14 wzc1 core6 v2 re-eval on .252 (2026-08-08 heartbeat)
# Purpose: (1) fill the last gap in the clean post-boundary single-arch Table 4
# ladder (Jul 28's keep14 wzc1 predates the Aug 2 zwfy6 driver-scp boundary and
# the `--save_per_example` commit; every other rung has an Aug 8 measurement);
# (2) independently test candidate (A) dataset drift by comparing today's wzc1
# keep14 core6 against Jul 28's 0.59376 — same disk, same tracked driver, but
# potentially different HF dataset cache state (zwfy6 HF cache has Aug 4 locks
# for winogrande / ai2_arc, the two tasks dominating flip counts).
#
# Minimal recipe: core6 downstream ONLY (no PPL, no MMLU-content, no closedbook)
# because both use cases need only core6. Runtime target ~5 min on 8× L20A.
#
# Byte-identical to `_run_olmo2_within_disk_floor_v3.sh` on the downstream leg:
# same driver, `chat_template=False` (default), `--save_per_example`,
# `--num_shards 8 --batch_size 8`, per-shard `LOCAL_RANK=0 RANK=$g`,
# `assert_8shards` before merge. Output name `_wzc1_v2` never overwrites any
# prior measurement.
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/hf_datasets_cache
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

CORE="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
BASE=../models/OLMo-2-1124-7B
CKPT=outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt
NAME=7B_keep14_step200000_wzc1_v2

# pre-flight
if [ ! -f "$CKPT" ]; then echo "FATAL: ckpt missing: $CKPT"; exit 2; fi
if [ ! -d "$BASE" ]; then echo "FATAL: base missing: $BASE"; exit 2; fi
if [ -d "olmo2_downstream_results/$NAME" ]; then
  echo "FATAL: output dir already exists (would clobber): olmo2_downstream_results/$NAME"; exit 2
fi
echo "[$(date '+%F %T')] pre-flight OK; NAME=$NAME"

assert_8shards () {
  local D=olmo2_downstream_results/$NAME
  local MISS=0
  for g in 0 1 2 3 4 5 6 7; do
    if [ ! -f "$D/per_example_hellaswag_shard${g}of8.jsonl" ]; then
      echo "SHARD MISSING: g=$g"; MISS=$((MISS+1))
    fi
  done
  [ $MISS -gt 0 ] && { echo "ABORT merge: $MISS/8 shards missing"; return 1; }
  return 0
}

echo "[$(date '+%F %T')] (1) prepare data"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE" \
    > logs/olmo2_downstream_${NAME}_prepare.log 2>&1 || { echo "prepare FAILED"; exit 3; }

echo "[$(date '+%F %T')] (2) 8-way shard on L20A"
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE" \
    --num_shards 8 --shard_index $g --batch_size 8 \
    --save_per_example \
    --output_name "$NAME" \
    > logs/olmo2_downstream_${NAME}_shard${g}.log 2>&1 &
done
wait

assert_8shards || exit 4

echo "[$(date '+%F %T')] (3) merge"
$PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
echo "[$(date '+%F %T')] SUMMARY:"
cat olmo2_downstream_results/$NAME/summary.json | head -60

# comparison — Jul 28 anchor was .59376
echo "[$(date '+%F %T')] === COMPARISON vs Jul 28 anchor (.59376) ==="
$PY - <<EOF
import json
CORE=['hellaswag','arc_challenge','arc_easy','piqa','openbookqa','winogrande']
new=json.load(open('olmo2_downstream_results/${NAME}/summary.json'))
old=json.load(open('olmo2_downstream_results/7B_keep14_step200000/summary.json'))
def core(s):
    t=s.get('tasks',s)
    return sum(t[k].get('acc' if k=='winogrande' else 'acc_norm') for k in CORE)/6
c_new=core(new); c_old=core(old)
print(f'  Jul28 core6 = {c_old:.5f}')
print(f'  Aug08 core6 = {c_new:.5f}')
print(f'  delta      = {(c_new-c_old)*100:+.4f} pp')
print(f'  ==> {"MATCH (data cache stable)" if abs(c_new-c_old)<1e-4 else "DIFFERENT — dataset drift candidate confirmed on wzc1"}')
EOF

echo "[$(date '+%F %T')] DONE"
