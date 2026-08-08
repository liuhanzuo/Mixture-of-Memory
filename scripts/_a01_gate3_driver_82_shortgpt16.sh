#!/usr/bin/env bash
# A01 gate-3 EXTENSION on .82: shortgpt16 dtype arm.
#
# WHY: shortgpt16 (structural pruning, healed 200k) is the arm that made B04's
# ladder n=6 with p=0.0028 -- it sits between base and keep14 on both core6 and
# margin distributions. Adding its (tie_rate, gap_median, letter_acc_delta)
# triple to the gate-3 side gives a full 6-rung dtype ladder that matches B04's
# rung set one-for-one. If gap_median falls monotonically with damage across
# base/shortgpt16/keep14/keep12/keep10/keep8, gate-3 and B04 are provably
# measuring the same compression from two directions.
set -u

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
NGPU=8
BS=16
# shortgpt16 is a plain 16-layer prune with 0 fresh layers, healed for 200k.
CKPT=outputs/olmo2_probe2_7B_shortgpt16/step200000.pt
KFL=16
NFL=0
ARM=7B_shortgpt16_step200000_dtype

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs results/a01_gate3/dtype_runs

SCRIPT=proposal/active/A01-null-calibration-methodology/code/a01_gate3_fp32_vs_bf16.py
BASE=../models/OLMo-2-1124-7B
PROGRESS=logs/a01_gate3_shortgpt16_progress.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

log "DRIVER START on $(hostname) -- shortgpt16 dtype arm, ngpu=$NGPU bs=$BS"

if [ -f "results/a01_gate3/dtype_runs/${ARM}/dtype_summary.json" ]; then
    log "SKIP -- dtype_summary.json already exists"; exit 0
fi
if [ ! -f "$CKPT" ]; then
    log "CKPT absent, trying alternates"
    for alt in outputs/olmo2_probe2_7B_shortgpt16fresh0/step200000.pt \
               outputs/olmo2_probe2_7B_shortgpt/step200000.pt; do
        if [ -f "$alt" ]; then CKPT="$alt"; log "using alt $CKPT"; break; fi
    done
fi
[ -f "$CKPT" ] || { log "FATAL -- no shortgpt16 ckpt on disk"; exit 1; }

for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES="$i" "$PY" "$SCRIPT" \
        --base_model "$BASE" --output_name "$ARM" \
        --num_shards "$NGPU" --shard_index "$i" --batch_size "$BS" \
        --ckpt "$CKPT" --keep_front_layers "$KFL" --n_fresh_layers "$NFL" \
        > "logs/a01_gate3_${ARM}_shard${i}.log" 2>&1 &
done
log "launched 8 shards (ckpt=$CKPT keep=$KFL fresh=$NFL)"
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
