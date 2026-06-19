#!/usr/bin/env bash
# Phase-3 S4b passkey GATING — eval a trained S4b checkpoint (LLaMA-1-7B + landmark
# mechanism, trained on RedPajama + learned block-gate) with the SAME protocol as S0:
#   n_garbage = 0/4000/8000/15000/30000/60000/115000  (~70/1.1k/2.2k/4k/8k/16k/32k tok)
#   50 tests/length, top_k=5. mem arm only (base already anchored in S0).
# 8-way GPU sharding on 本机: each GPU runs test indices [i::8], pooled by score script.
#
# Usage: CKPT=<path-to-ckpt-dir> CK_NAME=s4b_step1000 bash scripts/eval_landmark_S4b_passkey.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
EXT="$PROJECT_ROOT/external"
REPO="$EXT/landmark-attention/llama"
PY="$EXT/landmark_venv/bin/python"

CKPT="${CKPT:?set CKPT to the checkpoint dir}"
CK_NAME="${CK_NAME:-s4b_ckpt}"
OUTDIR="$EXT/landmark/results_s4b/$CK_NAME"
mkdir -p "$OUTDIR"

export LM_TUNED="$CKPT"
export LM_MODELS="mem"
export LM_TOPK="${LM_TOPK:-5}"
export LM_NTESTS="${LM_NTESTS:-50}"
export LM_NVALUES="${LM_NVALUES:-0,4000,8000,15000,30000,60000,115000}"
export LM_CACHE="$EXT/landmark/hf-cache"
export LM_NSHARDS=8
export LM_SEED="${LM_SEED:-1234}"

cd "$REPO"
pids=()
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s LM_SHARD_INDEX=$s LM_MEM_DEVICE=cuda:0 \
    LM_OUT="$OUTDIR/mem_shard${s}of8.csv" \
    "$PY" "$EXT/landmark/run_passkey.py" > "$OUTDIR/shard${s}.log" 2>&1 &
  pids+=($!)
done
echo "launched 8 shards for $CK_NAME, pids: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "shards done (fail=$fail)"

# pool shards: sum correct/total per n_garbage
"$PY" - "$OUTDIR" <<'PYEOF'
import csv, glob, os, sys
d = sys.argv[1]
agg = {}
toks = {}
for f in sorted(glob.glob(os.path.join(d, "mem_shard*of8.csv"))):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            n = int(r["n_garbage"])
            c = int(r["correct"]); t = int(r["num_tests"])
            agg.setdefault(n, [0,0]); agg[n][0]+=c; agg[n][1]+=t
            toks[n] = r["num_tokens"]
out = os.path.join(d, "pooled.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["n_garbage","num_tokens","correct","total","accuracy_pct"])
    for n in sorted(agg):
        c,t = agg[n]; acc = 100.0*c/t if t else 0.0
        w.writerow([n, toks.get(n,""), c, t, round(acc,1)])
        print(f"n={n} tok~{toks.get(n,'')}: {c}/{t} = {acc:.1f}%")
print("pooled ->", out)
PYEOF
