#!/usr/bin/env bash
# Re-run all 8 refinement cells (HE+ 4-tier + MBPP+ 4-tier) with the FIXED
# verifier (evalplus.eval.untrusted_check). Baselines are untouched -- they
# were graded by evalplus and were never affected by the sandbox bug.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1

run_refine () {
  local NAME="$1" INPUT="$2" DSET="$3"; shift 3
  local OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  if [ ! -s "$INPUT/solutions.jsonl" ]; then echo "$NAME: input $INPUT missing SKIP"; return; fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== $NAME (input=$INPUT dset=$DSET) ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/refine_verifier_guided.py \
        --input-run "$INPUT" \
        --dataset "$DSET" --policy restart \
        --refine-steps 256 --refine-temp 0.6 \
        --checkpoint models/Dream-Coder-v0-Instruct-7B \
        --output-dir "$OUT" --max-new-tokens 512 \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
  $PY -m evalplus.evaluate --dataset "$DSET" --samples "$OUT/solutions.jsonl" > "$OUT/evalplus.out" 2>&1
  grep -E "^pass@1|humaneval |mbpp " "$OUT/evalplus.out" | head -4
  $PY - "$OUT" <<'PYEOF'
import json, sys, glob, os
outdir=sys.argv[1]; rows=[]
for p in sorted(glob.glob(os.path.join(outdir,'metrics.rank*.jsonl'))):
  for l in open(p): rows.append(json.loads(l))
n=len(rows); kept=sum(1 for r in rows if r['action']=='keep')
rescued=sum(1 for r in rows if (not r['prior_ok']) and r['new_ok'])
lost=sum(1 for r in rows if r['prior_ok'] and (not r['new_ok']))
sec=sum(r['refine_seconds'] for r in rows)
print(f"  refine_stats: total={n} kept={kept} refined={n-kept} rescued={rescued} newly_lost={lost} sec={sec:.1f}")
PYEOF
}

# HE+ 4-tier
run_refine refine_tiny_restart_s256    runs/scaffold_tiny_heplus   humaneval
run_refine refine_small_restart_s256   runs/scaffold_small_heplus  humaneval
run_refine refine_medium_restart_s256  runs/scaffold_medium_heplus humaneval
run_refine refine_large_restart_s256   runs/scaffold_large_heplus  humaneval

# MBPP+ 4-tier
run_refine refine_tiny_restart_s256_mbpp    runs/scaffold_tiny_mbppplus    mbpp
run_refine refine_small_restart_s256_mbpp   runs/scaffold_small_mbppplus   mbpp
run_refine refine_medium_restart_s256_mbpp  runs/scaffold_medium_mbppplus  mbpp
run_refine refine_large_restart_s256_mbpp   runs/scaffold_large_mbppplus   mbpp

echo "[$(date '+%F %T')] ===== FIXED-verifier full refine grid DONE ====="
