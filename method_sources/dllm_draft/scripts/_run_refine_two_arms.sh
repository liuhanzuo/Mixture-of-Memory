#!/usr/bin/env bash
# Refinement two-arm study: on nfe256 baseline (55.5% HE+, 33 syntax-failed),
# apply restart vs remask, both at 256 refinement steps.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1

run_arm () {
  local NAME="$1" POLICY="$2"; shift 2
  local OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== $NAME ($POLICY) ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/refine_verifier_guided.py \
        --input-run runs/dream_instruct_heplus_nfe256 \
        --policy $POLICY --refine-steps 256 --refine-temp 0.6 \
        --checkpoint models/Dream-Coder-v0-Instruct-7B \
        --output-dir "$OUT" \
        "$@" \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
  $PY -m evalplus.evaluate --dataset humaneval --samples "$OUT/solutions.jsonl" \
      > "$OUT/evalplus.out" 2>&1
  grep -E "^pass@1|humaneval " "$OUT/evalplus.out" | head -4
  # per-task refine summary
  $PY - "$OUT" <<'PYEOF'
import json, sys, glob, os
outdir=sys.argv[1]
rows=[]
for p in sorted(glob.glob(os.path.join(outdir,'metrics.rank*.jsonl'))):
  for l in open(p): rows.append(json.loads(l))
n=len(rows); kept=sum(1 for r in rows if r['action']=='keep')
refined=n-kept
rescued=sum(1 for r in rows if (not r['prior_ok']) and r['new_ok'])
lost=sum(1 for r in rows if r['prior_ok'] and (not r['new_ok']))
sec=sum(r['refine_seconds'] for r in rows)
print(f"  refine_stats: total={n} kept={kept} refined={refined} rescued={rescued} newly_lost={lost} refine_seconds={sec:.1f}")
PYEOF
}

run_arm refine_nfe256_restart_s256 restart
run_arm refine_nfe256_remask50_s256 remask --remask-frac 0.5

echo "[$(date '+%F %T')] ===== refine two-arm DONE ====="
