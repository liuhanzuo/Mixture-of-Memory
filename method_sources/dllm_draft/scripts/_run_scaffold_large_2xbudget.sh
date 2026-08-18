#!/usr/bin/env bash
# Falsifiability test: is scaffold-Large's Medium-inferior score budget-truncation,
# not depth-harm? The current run uses --max-model-calls 512 and hits
# model_call_budget on 17 (HE+) / 35 (MBPP+) tasks. This rerun doubles the budget
# to 1024 to test whether those tasks resolve and Large catches Medium.
# Expected: Large@1024 pass@1 ~ Medium, and model_call_budget hits ~ 0.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

run_large () {
  local DS="$1" DF="$2" HEV="$3"
  local NAME="scaffold_large_${HEV}_budget1024"
  local OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  mkdir -p "$OUT"
  [[ "$DS" == "humaneval" ]] && export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl" || unset HUMANEVAL_OVERRIDE_PATH
  echo "[$(date '+%F %T')] ===== $NAME ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/generate_evalplus_scaffold.py \
        --checkpoint models/Scaffold-v0-stage1-7B \
        --dataset "$DS" \
        --data-file "$DF" \
        --output-dir "$OUT" \
        --max-model-calls 1024 --transfer-tokens 1 \
        --runtime-config-label large \
        --initial-root-slots 1 --initial-body-slots 2 \
        --initial-statement-masks 4 --initial-function-header-masks 4 \
        --initial-loop-header-masks 4 --initial-condition-masks 3 \
        --max-tree-depth 16 --max-lines-per-body 128 --max-total-lines 1024 \
        --max-tokens-per-hole 512 --max-expansions 512 \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
  $PY -m evalplus.evaluate --dataset "$DS" --samples "$OUT/solutions.jsonl" > "$OUT/evalplus.out" 2>&1
  grep -E "^pass@1|humaneval |mbpp " "$OUT/evalplus.out" | head -4
  $PY - "$OUT" <<'PYEOF'
import json, sys, glob, os, statistics
from collections import Counter
outdir=sys.argv[1]; rows=[]
for p in sorted(glob.glob(os.path.join(outdir,'metrics.rank*.jsonl'))):
  for l in open(p):
    try: rows.append(json.loads(l))
    except: pass
n=len(rows)
Cterm=Counter((r.get('process') or r.get('failure_process') or {}).get('termination_reason') for r in rows)
nfes=[(r.get('process') or {}).get('nfe') for r in rows]; nfes=[x for x in nfes if isinstance(x,(int,float))]
cmt=[(r.get('process') or {}).get('cumulative_model_tokens') for r in rows]; cmt=[x for x in cmt if isinstance(x,(int,float))]
print(f"  n={n} termination={dict(Cterm)}")
if nfes: print(f"  nfe mean={statistics.mean(nfes):.1f} median={statistics.median(nfes)} max={max(nfes)}")
if cmt: print(f"  cumulative_model_tokens mean={statistics.mean(cmt):.0f}")
PYEOF
}

run_large humaneval data/evalplus/humaneval_plus.jsonl heplus
run_large mbpp data/evalplus/mbpp_plus.jsonl mbppplus
echo "[$(date '+%F %T')] ===== large 2x budget falsifiability DONE ====="
