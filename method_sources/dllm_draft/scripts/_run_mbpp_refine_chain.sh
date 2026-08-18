#!/usr/bin/env bash
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1

# Stage 1: nfe256 MBPP+ baseline (weaker start, more room for refinement)
run_dream_mbpp () {
  local NAME="$1" STEPS="$2"; shift 2
  local OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== $NAME ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/generate_evalplus_dream.py \
        --checkpoint models/Dream-Coder-v0-Instruct-7B \
        --dataset mbpp --data-file data/evalplus/mbpp_plus.jsonl \
        --output-dir "$OUT" \
        --steps $STEPS --max-new-tokens 512 --temperature 0.1 --top-p 0.95 \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" | tail -2
  $PY -m evalplus.evaluate --dataset mbpp --samples "$OUT/solutions.jsonl" > "$OUT/evalplus.out" 2>&1
  grep -E "^pass@1|mbpp " "$OUT/evalplus.out" | head -4
}

run_refine_mbpp () {
  local NAME="$1" INPUT="$2" POLICY="$3"; shift 3
  local OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== $NAME ($POLICY, input=$INPUT) ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/refine_verifier_guided.py \
        --input-run "$INPUT" --policy $POLICY \
        --refine-steps 256 --refine-temp 0.6 \
        --checkpoint models/Dream-Coder-v0-Instruct-7B \
        --output-dir "$OUT" \
        --max-new-tokens 512 \
        "$@" \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" | tail -2
  $PY -m evalplus.evaluate --dataset mbpp --samples "$OUT/solutions.jsonl" > "$OUT/evalplus.out" 2>&1
  grep -E "^pass@1|mbpp " "$OUT/evalplus.out" | head -4
  $PY - "$OUT" <<'PYEOF'
import json, sys, glob, os
outdir=sys.argv[1]; rows=[]
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

run_dream_mbpp dream_instruct_mbppplus_nfe256 256
run_refine_mbpp refine_mbpp256_restart_s256 runs/dream_instruct_mbppplus_nfe256 restart

echo "[$(date '+%F %T')] ===== MBPP refine chain DONE ====="
