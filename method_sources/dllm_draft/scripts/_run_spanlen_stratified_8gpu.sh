#!/usr/bin/env bash
# Span-length stratified infilling audit: 3 arms x 2 splits, 8-way sharded on .73.
#
# Arms (all greedy, T=0, so runs are re-runnable/deterministic per arm):
#   dreamon_fim  DreamOn-v0-7B      native variable-length infilling  (NON-oracle)
#   dream_fim    Dream-Coder-Instr  fixed canvas + ORACLE middle length
#   qwen_fim     Qwen2.5-Coder-7B   native FIM sentinels              (NON-oracle)
#
# max_new_tokens=256 so the 179 MultiLine items with gt_len>128 are NOT budget-
# truncated. Truncation/abort counts are recorded per item by the generator and
# are reported separately from grading failures.
#
# Sharding contract: CUDA_VISIBLE_DEVICES=$g with LOCAL_RANK=0 and RANK=$g.
# Setting LOCAL_RANK=$g gives 'invalid device ordinal' on shards 1-7.
set -uo pipefail

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104
PY=$ROOT/.venv_dream/bin/python
export PYTHONPATH=$ROOT:$ROOT/scripts
cd "$ROOT" || exit 1
mkdir -p logs runs/spanlen

MAX_NEW=${MAX_NEW:-256}
SPLITS=${SPLITS:-"RandomSpan MultiLine"}
ARMS=${ARMS:-"qwen_fim dream_fim dreamon_fim"}

ck_for() {
  case "$1" in
    dreamon_fim|dreamon_oracle) echo models/DreamOn-v0-7B ;;
    dream_fim|dream_prefix)     echo models/Dream-Coder-v0-Instruct-7B ;;
    qwen_fim|qwen_prefix)       echo models/Qwen2.5-Coder-7B ;;
  esac
}

for SPLIT in $SPLITS; do
  SPLIT_FILE=$ROOT/data/infilling/spanlen_${SPLIT}.jsonl
  EXPECTED=$(wc -l < "$SPLIT_FILE")
  for ARM in $ARMS; do
    CK=$(ck_for "$ARM")
    DIR=$ROOT/runs/spanlen/${SPLIT}_${ARM}
    mkdir -p "$DIR/shards"
    if [ -f "$DIR/solutions.jsonl" ]; then
      echo "[$(date -Is)] SKIP $SPLIT/$ARM (solutions.jsonl exists)"
      continue
    fi
    echo "[$(date -Is)] START $SPLIT/$ARM ckpt=$CK n=$EXPECTED max_new=$MAX_NEW"
    pids=()
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY "$ROOT/scripts/generate_infilling.py" \
        --arm "$ARM" --checkpoint "$CK" \
        --data-file "$SPLIT_FILE" \
        --output-dir "$DIR/shards" \
        --max-new-tokens "$MAX_NEW" \
        --temperature 0.0 --resume \
        >"$ROOT/logs/spanlen_${SPLIT}_${ARM}_g${g}.log" 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "[$(date -Is)] GEN DONE $SPLIT/$ARM fail=$fail"

    # Merge with a HARD coverage assertion. A silent partial merge corrupts the
    # metric; that class of bug already caused a retraction in this project.
    $PY - "$DIR" "$EXPECTED" <<'PYX'
import json, sys, glob, os
d, expected = sys.argv[1], int(sys.argv[2])
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
sols = [r for p in sorted(glob.glob(d+"/shards/solutions.rank*.jsonl")) for r in rd(p)]
mets = [r for p in sorted(glob.glob(d+"/shards/metrics.rank*.jsonl")) for r in rd(p)]
nsh = len(glob.glob(d+"/shards/solutions.rank*.jsonl"))
assert nsh == 8, f"expected 8 shards, found {nsh}"
sols.sort(key=lambda r: r["task_id"]); mets.sort(key=lambda r: r["task_id"])
ids = {r["task_id"] for r in sols}
assert len(ids) == len(sols), f"duplicate task_ids: {len(sols)} rows, {len(ids)} unique"
assert len(sols) == expected, f"COVERAGE {len(sols)} != expected {expected}"
assert len(mets) == expected, f"COVERAGE metrics {len(mets)} != expected {expected}"
for name, rows in (("solutions.jsonl", sols), ("metrics.jsonl", mets)):
    with open(os.path.join(d, name), "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
print(json.dumps({"merged": len(sols), "shards": nsh, "expected": expected}))
PYX
    if [ $? -ne 0 ]; then echo "[$(date -Is)] MERGE/COVERAGE FAILED $SPLIT/$ARM"; rm -f "$DIR/solutions.jsonl"; continue; fi
    echo "[$(date -Is)] MERGED $SPLIT/$ARM"
  done
done
echo "[$(date -Is)] ALL DONE"
