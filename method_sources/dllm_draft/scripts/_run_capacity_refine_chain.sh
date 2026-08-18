#!/usr/bin/env bash
# Killer experiment: does verifier-guided refinement salvage low-capacity failures?
# If YES on Tiny/Small: "low structural capacity + refinement" Pareto-dominates
#   Medium alone -> small runtime + refine is enough
# If NO: structural capacity is genuinely necessary, refinement can't substitute
#   -> capacity floor and refinement are ORTHOGONAL axes (also a good result)
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1

run_refine () {
  local NAME="$1" INPUT="$2"; shift 2
  local OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== $NAME (input=$INPUT) ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/refine_verifier_guided.py \
        --input-run "$INPUT" \
        --dataset humaneval --policy restart \
        --refine-steps 256 --refine-temp 0.6 \
        --checkpoint models/Dream-Coder-v0-Instruct-7B \
        --output-dir "$OUT" --max-new-tokens 512 \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
  $PY -m evalplus.evaluate --dataset humaneval --samples "$OUT/solutions.jsonl" > "$OUT/evalplus.out" 2>&1
  grep -E "^pass@1|humaneval " "$OUT/evalplus.out" | head -4
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

run_refine refine_tiny_restart_s256   runs/scaffold_tiny_heplus
run_refine refine_small_restart_s256  runs/scaffold_small_heplus
run_refine refine_large_restart_s256  runs/scaffold_large_heplus

echo "[$(date '+%F %T')] ===== capacity-refine chain DONE ====="
