#!/usr/bin/env bash
# 8-GPU sharded k-span infilling generation. One arm per invocation.
#
#   ARM=diffusion CKPT=models/Dream-Coder-v0-Instruct-7B PYBIN=... bash scripts/_run_kspan_8gpu.sh
#
# Per-shard sharding on this cluster REQUIRES LOCAL_RANK=0 together with
# CUDA_VISIBLE_DEVICES=$g: after CUDA_VISIBLE_DEVICES torch only sees one card,
# so LOCAL_RANK=$g would raise `invalid device ordinal` on shards 1-7.
set -uo pipefail

ARM="${ARM:?set ARM}"
CKPT="${CKPT:-}"
SPEC="${SPEC:-data/kspan/kspan_spec_v1.jsonl}"
OUTDIR="${OUTDIR:-runs/kspan_${ARM}}"
PYBIN="${PYBIN:?set PYBIN}"
KS="${KS:-}"
EXTRA="${EXTRA:-}"
NGPU="${NGPU:-8}"
LOGTAG="${LOGTAG:-kspan_${ARM}}"

mkdir -p "$OUTDIR" logs
echo "arm=$ARM spec=$SPEC out=$OUTDIR ks=${KS:-all} ngpu=$NGPU"
md5sum "$SPEC"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g WORLD_SIZE=$NGPU \
  "$PYBIN" scripts/generate_kspan.py \
      --arm "$ARM" \
      --spec "$SPEC" \
      ${CKPT:+--checkpoint "$CKPT"} \
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

echo "--- shard row counts (must sum to spec size; a silent partial merge has destroyed a measurement before) ---"
wc -l "$OUTDIR"/solutions.rank*.jsonl 2>/dev/null | tail -1
for g in $(seq 0 $((NGPU-1))); do
  printf 'rank%02d %s\n' "$g" "$(wc -l < "$OUTDIR/solutions.rank0${g}.jsonl" 2>/dev/null || echo MISSING)"
done
exit $fail
