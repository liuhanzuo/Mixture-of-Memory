#!/usr/bin/env bash
# ==============================================================================
# Paper-faithful baseline reproduction on 8× L20A (sm_100) for three checkpoints
# on HumanEval+ (n=164) and MBPP+ (n=378) using EvalPlus v0.1.10 / v0.2.0.
#
# Rationale: sanity-check the pipeline end-to-end BEFORE any elastic-scaffold or
# subtree-collapse experiment, and produce clean numbers that (a) match the
# Dream-Coder v0 technical report Table 1/4 for Base+Instruct, (b) become the
# reference row for future ablations. DreamOn HE+/MBPP+ are not published so
# these are novel measurements.
#
# Sampler recipes from vendor/Dream-Coder/base/eval_code_base.sh and the
# Instruct configs cited in the arxiv 2509.01142 report.
#
# Env: /opt/conda/envs/dllm-env (torch 2.11+cu128, transformers 4.51.3).
# Model shards: 8 GPUs, tasks split by index % world_size.
# Post-process: merge_evalplus_shards.py + evalplus.evaluate for pass@1.
# ==============================================================================
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1
mkdir -p logs runs

run_one () {
  local NAME="$1" SCRIPT="$2" CKPT="$3" DATASET="$4" DATAFILE="$5"; shift 5
  local OUTDIR="runs/$NAME"
  if [ -s "$OUTDIR/solutions.jsonl" ]; then
    echo "[$(date '+%F %T')] $NAME: solutions.jsonl exists -> SKIP"
    return
  fi
  mkdir -p "$OUTDIR"
  echo "[$(date '+%F %T')] ===== $NAME ====="
  local NG=8
  for g in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=$NG \
      $PY -u "$SCRIPT" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --data-file "$DATAFILE" \
        --output-dir "$OUTDIR" \
        "$@" \
        > "$OUTDIR/shard${g}.log" 2>&1 &
  done
  wait
  # merge per-rank solutions into one file evalplus can grade
  $PY - "$OUTDIR" <<'PYEOF'
import json, sys, os, glob
outdir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(outdir, 'solutions.rank*.jsonl')))
rows = []
for p in paths:
    for l in open(p):
        rows.append(json.loads(l))
seen = set()
uniq = []
for r in rows:
    if r['task_id'] in seen: continue
    seen.add(r['task_id']); uniq.append(r)
with open(os.path.join(outdir, 'solutions.jsonl'), 'w') as f:
    for r in uniq:
        f.write(json.dumps(r)+'\n')
print(f"merged {len(uniq)} solutions from {len(paths)} shards")
PYEOF
  ls "$OUTDIR"/*.jsonl 2>/dev/null | tail -3
  echo "[$(date '+%F %T')] $NAME done"
}

# ----------------- Dream-Coder-Instruct-7B HE+ (paper-faithful) --------------
# Sampler: temperature 0.1 top_p 0.95 steps 512, chat template ON.
# Paper claim: HumanEval 82.9 / MBPP 79.6 (Instruct Table 4).
run_one \
  "dream_coder_instruct_heplus" \
  "scripts/generate_evalplus_dream.py" \
  "models/Dream-Coder-v0-Instruct-7B" \
  "humaneval" \
  "data/evalplus/humaneval_plus.jsonl" \
  --steps 512 --max-new-tokens 512 --temperature 0.1

# ----------------- Dream-Coder-Instruct-7B MBPP+ ------------------------------
run_one \
  "dream_coder_instruct_mbppplus" \
  "scripts/generate_evalplus_dream.py" \
  "models/Dream-Coder-v0-Instruct-7B" \
  "mbpp" \
  "data/evalplus/mbpp_plus.jsonl" \
  --steps 512 --max-new-tokens 512 --temperature 0.1

# ----------------- DreamOn-v0-7B HE+ (novel measurement) ----------------------
# DreamOn does not publish HumanEval+/MBPP+. This run establishes them.
run_one \
  "dreamon_heplus" \
  "scripts/generate_evalplus_dreamon.py" \
  "models/DreamOn-v0-7B" \
  "humaneval" \
  "data/evalplus/humaneval_plus.jsonl" \
  --initial-masks 8 --max-new-tokens 512 --transfer-tokens 1

# ----------------- DreamOn-v0-7B MBPP+ ---------------------------------------
run_one \
  "dreamon_mbppplus" \
  "scripts/generate_evalplus_dreamon.py" \
  "models/DreamOn-v0-7B" \
  "mbpp" \
  "data/evalplus/mbpp_plus.jsonl" \
  --initial-masks 8 --max-new-tokens 512 --transfer-tokens 1

echo "[$(date '+%F %T')] ===== ALL 4 RUNS DONE ====="
