#!/usr/bin/env bash
# 8-GPU sharded k-span infilling generation, NON-ORACLE DreamOn arm.
#
# Per-shard sharding on this cluster REQUIRES LOCAL_RANK=0 together with
# CUDA_VISIBLE_DEVICES=$g: after CUDA_VISIBLE_DEVICES torch only sees one card,
# so LOCAL_RANK=$g would raise `invalid device ordinal` on shards 1-7.
#
# Usage:
#   CKPT=models/DreamOn-v0-7B PYBIN=/path/to/python bash scripts/_run_kspan_dreamon_nonoracle_8gpu.sh
set -uo pipefail

CKPT="${CKPT:-models/DreamOn-v0-7B}"
SPEC="${SPEC:-data/kspan/kspan_spec_v1.jsonl}"
OUTDIR="${OUTDIR:-runs/kspan_diffusion_nonoracle}"
PYBIN="${PYBIN:?set PYBIN}"
KS="${KS:-}"
EXTRA="${EXTRA:-}"
NGPU="${NGPU:-8}"
LOGTAG="${LOGTAG:-kspan_dreamon_nonoracle}"

mkdir -p "$OUTDIR" logs
echo "arm=dreamon_nonoracle spec=$SPEC out=$OUTDIR ks=${KS:-all} ngpu=$NGPU ckpt=$CKPT"
md5sum "$SPEC"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g WORLD_SIZE=$NGPU \
  "$PYBIN" scripts/generate_kspan_dreamon_nonoracle.py \
      --spec "$SPEC" \
      --checkpoint "$CKPT" \
      --output-dir "$OUTDIR" \
      ${KS:+--ks "$KS"} \
      $EXTRA \
      > "logs/${LOGTAG}_g${g}.log" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "SHARD $i FAILED (see logs/${LOGTAG}_g${i}.log)"
    fail=1
  fi
done

echo "--- shard row counts (must sum to spec size; a silent partial merge has destroyed a口径 before) ---"
wc -l "$OUTDIR"/solutions.rank*.jsonl 2>/dev/null | tail -1
for g in $(seq 0 $((NGPU-1))); do
  printf 'rank%02d %s\n' "$g" "$(wc -l < "$OUTDIR/solutions.rank0${g}.jsonl" 2>/dev/null || echo MISSING)"
done
exit $fail
