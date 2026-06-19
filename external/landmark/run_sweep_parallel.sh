#!/usr/bin/env bash
# Phase 1 S0 full passkey sweep, parallelized across 8 H20 GPUs.
# mem arm is the slow one (chunked landmark inference over long ctx) -> one length per GPU.
# base arm collapses early and is cheap -> all lengths on one GPU.
#
# n_garbage CHARS chosen so the garbage text (~3.7 chars/token) maps to target TOKEN lengths:
#   chars   ~tokens
#   0        ~70   (control)
#   4000     ~1.1k
#   8000     ~2.2k
#   15000    ~4k
#   30000    ~8k
#   60000    ~16k
#   115000   ~31k  (near garbage_inf cap 180k)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$(cd "$HERE/.." && pwd)"
REPO="$EXT/landmark-attention/llama"
PY="$EXT/landmark_venv/bin/python"
CKPTS="$EXT/landmark_ckpts"
export LM_BASE="$CKPTS/llama1_7b_base"
export LM_TUNED="$CKPTS/landmark_tuned"
export LM_CACHE="$HERE/hf-cache"
export LM_TOPK="${LM_TOPK:-5}"
export LM_NTESTS="${LM_NTESTS:-50}"
export LM_REPO="$REPO"
OUTDIR="$HERE/results"
mkdir -p "$OUTDIR"

NVALS=(0 4000 8000 15000 30000 60000 115000)

# --- mem arm: one length per GPU (GPU 0..6) ---
pids=()
for i in "${!NVALS[@]}"; do
  n="${NVALS[$i]}"
  gpu="$i"
  LM_MODELS=mem LM_MEM_DEVICE="cuda:0" CUDA_VISIBLE_DEVICES="$gpu" \
    LM_NVALUES="$n" LM_OUT="$OUTDIR/mem_n${n}.csv" \
    "$PY" "$HERE/run_passkey.py" > "$OUTDIR/mem_n${n}.log" 2>&1 &
  pids+=($!)
done

# --- base arm: all lengths on GPU 7 ---
LM_MODELS=base LM_BASE_DEVICE="cuda:0" CUDA_VISIBLE_DEVICES="7" \
  LM_NVALUES="$(IFS=,; echo "${NVALS[*]}")" LM_OUT="$OUTDIR/base_all.csv" \
  "$PY" "$HERE/run_passkey.py" > "$OUTDIR/base_all.log" 2>&1 &
pids+=($!)

echo "[sweep] launched ${#pids[@]} jobs, waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

# --- merge ---
{
  echo "model,n_garbage,num_tokens,num_tests,correct,accuracy_pct"
  for f in "$OUTDIR"/mem_n*.csv "$OUTDIR"/base_all.csv; do
    [ -f "$f" ] && tail -n +2 "$f"
  done
} > "$OUTDIR/passkey_full.csv"
echo "[sweep] done (fail=$fail) -> $OUTDIR/passkey_full.csv"
cat "$OUTDIR/passkey_full.csv"
