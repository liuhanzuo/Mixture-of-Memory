#!/usr/bin/env bash
# A01 gate-3 EXTENSION on .82: keep10 dtype arm.
#
# WHY: gate-3 base + keep14 + keep12 + keep8 give four rungs of the same
# (tie_rate, gap_median, letter_acc_delta) triple that connects gate-3 to B04's
# six-rung per-item margin ladder. Adding keep10 rounds the gate-3 ladder to
# FIVE rungs -- one for one with B04's five keepN rungs (base + keep{14,12,10,8}),
# which is what lets a downstream analysis regress gap_median on core6 across
# BOTH gates on the same rung set.
#
# .73 is currently running the keep14 + keep12 dtype arms (do NOT collide).
# .82 has zero contention; this is the fastest way to complete the ladder.
#
# ONE-ARM ONLY, no skip logic needed (there is no summary.json yet).
set -u

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
NGPU=8
BS=16
CKPT=outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt
KFL=10
NFL=2
ARM=7B_keep10_step83500_dtype

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs results/a01_gate3/dtype_runs

SCRIPT=proposal/active/A01-null-calibration-methodology/code/a01_gate3_fp32_vs_bf16.py
BASE=../models/OLMo-2-1124-7B
PROGRESS=logs/a01_gate3_keep10_progress.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

log "DRIVER START on $(hostname) -- keep10 dtype arm, ngpu=$NGPU bs=$BS"

if [ -f "results/a01_gate3/dtype_runs/${ARM}/dtype_summary.json" ]; then
    log "SKIP -- dtype_summary.json already exists"
    exit 0
fi
if [ ! -f "$CKPT" ]; then
    log "FATAL -- ckpt absent: $CKPT"
    exit 1
fi

for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES="$i" "$PY" "$SCRIPT" \
        --base_model "$BASE" \
        --output_name "$ARM" \
        --num_shards "$NGPU" --shard_index "$i" \
        --batch_size "$BS" \
        --ckpt "$CKPT" --keep_front_layers "$KFL" --n_fresh_layers "$NFL" \
        > "logs/a01_gate3_${ARM}_shard${i}.log" 2>&1 &
done
log "launched 8 shards"
wait
log "all shards done, merging"

"$PY" "$SCRIPT" --merge --output_name "$ARM" --num_shards "$NGPU" \
    --base_model "$BASE" --ckpt "$CKPT" \
    --keep_front_layers "$KFL" --n_fresh_layers "$NFL" \
    >> "logs/a01_gate3_${ARM}_merge.log" 2>&1

if [ -f "results/a01_gate3/dtype_runs/${ARM}/dtype_summary.json" ]; then
    log "MERGE OK -> results/a01_gate3/dtype_runs/${ARM}/dtype_summary.json"
else
    log "MERGE FAILED"; exit 1
fi
