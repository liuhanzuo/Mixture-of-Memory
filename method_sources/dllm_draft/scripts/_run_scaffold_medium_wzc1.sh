#!/usr/bin/env bash
# Scaffold-Coder stage1 (global_step_4465) Medium runtime on HumanEval+, 8 GPUs.
# Uses the freshly synced ckpt at models/Scaffold-v0-stage1-7B.
# Medium capacity params from run_semantic_scaffold_medium_he_8gpu.sh (the config
# that RUNTIME_CAPACITY_SCREEN_RESULTS.md found eliminated all generation failures).
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"

OUT=runs/scaffold_medium_heplus
mkdir -p "$OUT"
if [ -s "$OUT/solutions.jsonl" ]; then echo "exists SKIP"; exit 0; fi
echo "[$(date '+%F %T')] ===== scaffold_medium_heplus (8 GPU) ====="
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
    $PY -u scripts/generate_evalplus_scaffold.py \
      --checkpoint models/Scaffold-v0-stage1-7B \
      --dataset humaneval \
      --data-file data/evalplus/humaneval_plus.jsonl \
      --output-dir "$OUT" \
      --max-model-calls 512 --transfer-tokens 1 \
      --runtime-config-label semantic-medium \
      --initial-root-slots 1 --initial-body-slots 2 \
      --initial-statement-masks 4 --initial-function-header-masks 4 \
      --initial-loop-header-masks 4 --initial-condition-masks 3 \
      --max-tree-depth 4 --max-lines-per-body 16 --max-total-lines 64 \
      --max-tokens-per-hole 32 --max-expansions 256 \
      > "$OUT/shard${g}.log" 2>&1 &
done
wait
$PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
$PY -m evalplus.evaluate --dataset humaneval --samples "$OUT/solutions.jsonl" \
    > "$OUT/evalplus.out" 2>&1
grep -E "^pass@1|humaneval " "$OUT/evalplus.out" | head -4
# capacity-pressure instrumentation summary
$PY - "$OUT" <<'PYEOF'
import json, sys, glob, os
from collections import Counter
outdir=sys.argv[1]; rows=[]
for p in sorted(glob.glob(os.path.join(outdir,'metrics.rank*.jsonl'))):
  for l in open(p):
    try: rows.append(json.loads(l))
    except: pass
n=len(rows)
print(f"  scaffold_stats: n={n}")
# collect failure reasons if present
C=Counter()
for r in rows:
  pr=r.get('process') or {}
  for k in ('failure_reason','termination','depth_capacity_hits','line_capacity_hits','token_capacity_hits'):
    if k in pr and pr[k]: C[f"{k}={pr[k]}"] += 1
  if r.get('error'): C['error']+=1
for k,v in C.most_common(12): print(f"    {k}: {v}")
nfes=[ (r.get('process') or {}).get('nfe') for r in rows ]
nfes=[x for x in nfes if isinstance(x,(int,float))]
if nfes:
  import statistics
  print(f"    nfe: mean={statistics.mean(nfes):.1f} median={statistics.median(nfes)} max={max(nfes)}")
PYEOF
echo "[$(date '+%F %T')] ===== scaffold baseline DONE ====="
