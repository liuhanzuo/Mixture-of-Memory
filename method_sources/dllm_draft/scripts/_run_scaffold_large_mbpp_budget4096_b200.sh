#!/usr/bin/env bash
# Decisive budget-saturation test for scaffold-Large on MBPP+ (runs on .252 / 8 GPU).
#
# Motivation. The @512 -> @1024 doubling on HE+ moved model_call_budget
# truncations 17 -> 17 (the *same* task set) and pass@1 .177 -> .177, i.e. exactly
# zero change. Generation is greedy (temperature=0.0, transfer_tokens=1), and we
# verified empirically that non-truncated solutions are byte-identical across the
# two budgets (HE+ 147/147, MBPP+ 53/53). Therefore raising --max-model-calls can
# ONLY change the outcome of tasks that actually hit the budget; every other task
# is bit-reproducible. On MBPP+@512 that truncated set is exactly 35 tasks, and
# every *resolved* MBPP+ task used at most nfe=190 of its 512 allowance.
#
# So instead of re-deriving the 343 already-deterministic tasks, this script pushes
# the 35 truncated tasks to an 8x budget (4096). Two possible outcomes, both decisive:
#   - some tasks now resolve  -> Large *was* partly budget-limited; quantifies how much
#   - all 35 still truncate   -> Large's deficit is non-terminating structural recursion,
#                               not a too-small call budget (falsifies the budget story
#                               on a second benchmark, independent of HE+)
# Merging these 35 back over the @512 run reconstructs a faithful full 378-task
# MBPP+ Large@4096 score, graded by the official evalplus grader.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=$ROOT/.venv_b200/bin/python
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/scripts:$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

NAME=scaffold_large_mbppplus_budget4096_hit35
OUT=runs/$NAME
mkdir -p "$OUT"
unset HUMANEVAL_OVERRIDE_PATH

echo "[$(date '+%F %T')] ===== $NAME (8x budget on the 35 truncated tasks) ====="
for g in 0 1 2 3 4 5 6 7; do
  # NOTE: with CUDA_VISIBLE_DEVICES=$g the process sees exactly one GPU, so
  # LOCAL_RANK must be 0 (it indexes cuda:LOCAL_RANK). RANK=$g does the logical
  # sharding (tasks[i] where i%8==RANK). Setting LOCAL_RANK=$g here would make
  # shards 1-7 die with "invalid device ordinal".
  CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
    $PY -u scripts/generate_evalplus_scaffold.py \
      --checkpoint models/Scaffold-v0-stage1-7B \
      --dataset mbpp \
      --data-file data/evalplus/mbpp_plus_largehit35.jsonl \
      --output-dir "$OUT" \
      --max-model-calls 4096 --transfer-tokens 1 \
      --runtime-config-label large \
      --initial-root-slots 1 --initial-body-slots 2 \
      --initial-statement-masks 4 --initial-function-header-masks 4 \
      --initial-loop-header-masks 4 --initial-condition-masks 3 \
      --max-tree-depth 16 --max-lines-per-body 128 --max-total-lines 1024 \
      --max-tokens-per-hole 512 --max-expansions 512 \
      > "$OUT/shard${g}.log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] generation done"

# Fail loudly if any shard died: all 8 metrics files must exist and cover 35 tasks.
$PY - "$OUT" <<'PYEOF'
import json, sys, glob, os
outdir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(outdir, 'metrics.rank*.jsonl')))
ids = set()
for p in paths:
    for l in open(p):
        try: ids.add(json.loads(l)['task_id'])
        except Exception: pass
print(f"  shards={len(paths)}/8 tasks={len(ids)}/35")
if len(paths) != 8 or len(ids) != 35:
    print("  SHARD INCOMPLETE -- refusing to report a score"); sys.exit(1)
PYEOF
[ $? -ne 0 ] && { echo "ABORT: incomplete shards"; exit 1; }

echo "[$(date '+%F %T')] ===== analysis ====="
$PY - "$OUT" <<'PYEOF'
import json, sys, glob, os, statistics
from collections import Counter
outdir = sys.argv[1]

def load(pattern, run):
    d = {}
    for p in sorted(glob.glob(os.path.join('runs', run, pattern))):
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            d[r['task_id']] = r
    return d

new_m = load('metrics.rank*.jsonl', os.path.basename(outdir))
old_m = load('metrics.rank*.jsonl', 'scaffold_large_mbppplus')

def term(r):
    return (r.get('process') or r.get('failure_process') or {}).get('termination_reason')

C = Counter(term(r) for r in new_m.values())
print(f"  35 previously-truncated tasks @4096: {dict(C)}")
newly = [t for t, r in new_m.items() if term(r) != 'model_call_budget']
print(f"  newly escaping the budget: {len(newly)}/35 {sorted(newly)}")

# cost accounting MUST include failure_process, else truncated (most expensive) tasks vanish
def cmt(r):
    return (r.get('process') or r.get('failure_process') or {}).get('cumulative_model_tokens')
v = [cmt(r) for r in new_m.values() if isinstance(cmt(r), (int, float))]
if v:
    print(f"  hit35 cumulative_model_tokens mean={statistics.mean(v):.0f} max={max(v)}")

# full-378 reconstruction: @4096 for the 35, deterministic @512 result for the other 343
merged = dict(old_m); merged.update(new_m)
Cf = Counter(term(r) for r in merged.values())
print(f"  reconstructed full MBPP+ n={len(merged)} termination={dict(Cf)}")
vf = [cmt(r) for r in merged.values() if isinstance(cmt(r), (int, float))]
if vf:
    print(f"  full-378 cumulative_model_tokens mean={statistics.mean(vf):.0f} (incl failure_process)")
PYEOF

# Build the reconstructed 378-task solution set and grade with the OFFICIAL grader.
$PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -1
MERGED=runs/${NAME}_full378
mkdir -p "$MERGED"
$PY - "$OUT" "$MERGED" <<'PYEOF'
import json, sys, glob, os
new_dir, merged_dir = sys.argv[1], sys.argv[2]
def load(d, pat):
    o = {}
    for p in sorted(glob.glob(os.path.join(d, pat))):
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            o[r['task_id']] = r
    return o
new = load(new_dir, 'solutions.rank*.jsonl')
old = load('runs/scaffold_large_mbppplus', 'solutions.rank*.jsonl')
out = dict(old); out.update(new)          # 8x-budget results override the 35
assert len(out) == 378, f"expected 378 got {len(out)}"
with open(os.path.join(merged_dir, 'solutions.jsonl'), 'w') as f:
    for k in sorted(out, key=lambda s: int(s.split('/')[1])):
        f.write(json.dumps(out[k]) + '\n')
print(f"  reconstructed {len(out)} solutions ({len(new)} from @4096) -> {merged_dir}/solutions.jsonl")
PYEOF

$PY -m evalplus.evaluate --dataset mbpp --samples "$MERGED/solutions.jsonl" > "$MERGED/evalplus.out" 2>&1
grep -E "^pass@1|^mbpp" "$MERGED/evalplus.out"
echo "[$(date '+%F %T')] ===== DONE ====="
