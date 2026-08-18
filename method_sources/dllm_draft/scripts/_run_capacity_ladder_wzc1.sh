#!/usr/bin/env bash
# Full HumanEval+ n=164 capacity ladder: Tiny / Small / Large (Medium already done)
# Extends RUNTIME_CAPACITY_SCREEN_RESULTS.md (n=64) to full statistical power on n=164.
# Params from vendor run_runtime_capacity_screen_8gpu.sh.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"

run_capacity () {
  local NAME="$1"; shift
  local OUT="runs/scaffold_${NAME}_heplus"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== scaffold_${NAME}_heplus (8 GPU) ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/generate_evalplus_scaffold.py \
        --checkpoint models/Scaffold-v0-stage1-7B \
        --dataset humaneval \
        --data-file data/evalplus/humaneval_plus.jsonl \
        --output-dir "$OUT" \
        --max-model-calls 512 --transfer-tokens 1 \
        --runtime-config-label "$NAME" \
        "$@" \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
  $PY -m evalplus.evaluate --dataset humaneval --samples "$OUT/solutions.jsonl" \
      > "$OUT/evalplus.out" 2>&1
  grep -E "^pass@1|humaneval " "$OUT/evalplus.out" | head -4
  $PY - "$OUT" <<'PYEOF'
import json, sys, glob, os, statistics
from collections import Counter
outdir=sys.argv[1]; rows=[]
for p in sorted(glob.glob(os.path.join(outdir,'metrics.rank*.jsonl'))):
  for l in open(p):
    try: rows.append(json.loads(l))
    except: pass
n=len(rows)
nfes=[(r.get('process') or {}).get('nfe') for r in rows]; nfes=[x for x in nfes if isinstance(x,(int,float))]
succ=sum(1 for r in rows if r.get('process') and not r.get('error'))
err=sum(1 for r in rows if r.get('error'))
C=Counter()
for r in rows:
  pr=r.get('process') or r.get('failure_process') or {}
  if pr.get('depth_capacity_hits'): C['depth_hit']+=1
  if pr.get('line_capacity_hits'): C['line_hit']+=1
  if pr.get('token_capacity_hits'): C['token_hit']+=1
  if pr.get('expand_budget_hits'): C['expand_budget']+=1
print(f"  n={n} generation_err={err} capacity_hits={dict(C)}")
if nfes:
  print(f"  nfe: mean={statistics.mean(nfes):.1f} median={statistics.median(nfes)} max={max(nfes)}")
PYEOF
}

run_capacity tiny \
  --initial-root-slots 2 --initial-body-slots 2 \
  --initial-statement-masks 1 --initial-function-header-masks 1 \
  --initial-loop-header-masks 1 --initial-condition-masks 1 \
  --max-tree-depth 1 --max-lines-per-body 2 --max-total-lines 16 \
  --max-tokens-per-hole 2 --max-expansions 32 --no-module-expand

run_capacity small \
  --initial-root-slots 1 --initial-body-slots 2 \
  --initial-statement-masks 2 --initial-function-header-masks 2 \
  --initial-loop-header-masks 2 --initial-condition-masks 2 \
  --max-tree-depth 2 --max-lines-per-body 4 --max-total-lines 32 \
  --max-tokens-per-hole 8 --max-expansions 128

run_capacity large \
  --initial-root-slots 1 --initial-body-slots 2 \
  --initial-statement-masks 4 --initial-function-header-masks 4 \
  --initial-loop-header-masks 4 --initial-condition-masks 3 \
  --max-tree-depth 16 --max-lines-per-body 128 --max-total-lines 1024 \
  --max-tokens-per-hole 512 --max-expansions 512

echo "[$(date '+%F %T')] ===== capacity ladder DONE ====="
