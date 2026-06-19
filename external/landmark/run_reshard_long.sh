#!/usr/bin/env bash
# Re-run the two slowest mem lengths (n60000 ~16k tok, n115000 ~30k tok) sharded
# across free GPUs to cut wall time. Each (length) split into 3 shards of ~17 tests.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$(cd "$HERE/.." && pwd)"
REPO="$EXT/landmark-attention/llama"
PY="$EXT/landmark_venv/bin/python"
CKPTS="$EXT/landmark_ckpts"
export LM_TUNED="$CKPTS/landmark_tuned"
export LM_CACHE="$HERE/hf-cache"
export LM_TOPK=5
export LM_NTESTS=50
export LM_REPO="$REPO"
export LM_MODELS=mem
export LM_MEM_DEVICE="cuda:0"
OUTDIR="$HERE/results"
mkdir -p "$OUTDIR"

# (n_garbage, gpu) shard map: n60000 -> GPUs 1,2,3 ; n115000 -> GPUs 5,6,7
declare -A MAP
launch() {  # $1=n $2=gpu $3=shard_index $4=nshards
  local n=$1 gpu=$2 si=$3 ns=$4
  CUDA_VISIBLE_DEVICES="$gpu" LM_NSHARDS="$ns" LM_SHARD_INDEX="$si" \
    LM_NVALUES="$n" LM_OUT="$OUTDIR/mem_n${n}_shard${si}of${ns}.csv" \
    "$PY" "$HERE/run_passkey.py" > "$OUTDIR/mem_n${n}_shard${si}of${ns}.log" 2>&1 &
  echo $!
}

pids=()
for si in 0 1 2; do pids+=($(launch 60000  $((1+si)) $si 3)); done
for si in 0 1 2; do pids+=($(launch 115000 $((5+si)) $si 3)); done

echo "[reshard] launched ${#pids[@]} jobs across GPUs 1,2,3,5,6,7"
for p in "${pids[@]}"; do wait "$p"; done
echo "[reshard] all done"
