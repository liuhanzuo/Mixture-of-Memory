#!/usr/bin/env bash
# B04 cross-family kill test -- the Qwen3-8B prune-heal ladder, bs16, core6.
#
# WHY THIS EXISTS
# ---------------
# B04 (eval-fragility / direction A) is established at maximum possible
# significance WITHIN OLMo-2-7B: over six rungs, Spearman(core6, median_margin)
# = +1.0000 and Spearman(core6, frac<0.005) = -1.0000, both at the n=6
# exact-permutation lower bound p = 0.0028. Its own verdict states the limit
# plainly: "NOT established beyond OLMo-2-7B. Cross-family replication (Qwen
# prune-heal ladder) is the next kill test."
#
# This is that test. Outcomes:
#   * rho stays near -1 (frac<0.005) / +1 (median_margin) at n>=5
#         -> the "structural damage compresses per-item decision margins" claim
#            is cross-family, and B04 becomes promotable to paper<X>
#            (only the CPU novelty check would remain)
#   * rho flips sign or goes flat
#         -> the effect is OLMo-2-specific and B04's headline must be narrowed
#            to a single-model observation. That is a real kill of the general claim.
# Either way MAIN's writeup changes, which is what makes this worth 8 GPUs.
#
# NODE: .21 (8x L20A 183GB, wzc1). Every asset is on wzc1 -- zero cross-disk cost.
#   Verified present before writing this script:
#     outputs/qwen3_minarch_armB_f12k2/final.pt          15G  step=2000    14L (keep12+fresh2)
#     outputs/qwen3_minarch_armB_f12k2_20k/final.pt      15G  step=20000   14L
#     outputs/qwen3_minarch_armB_f12k2_200k/final.pt     45G  step=200000  14L (has optimizer_state; loader reads model_state only)
#     outputs/qwen3_minarch_armB_f12k4/final.pt          17G  step=2000    16L (keep12+fresh4)
#     outputs/qwen3_minarch_scratch_f12k2/final.pt       15G  step=2000    14L from_scratch=True
#     ../models/Qwen--Qwen3-8b                           5 safetensors shards (full 36L base)
#
# THE ONE NON-OBVIOUS CORRECTNESS ISSUE
# -------------------------------------
# `scripts/eval_qwen3_probe2_downstream.py --save_per_example` writes
# `option_scores` + `acc_norm_score` but NOT `norm_lens` / `norm_scores`; that
# field pair was added only to the OLMo harness. B04's analyzer prefers
# `norm_scores` and SILENTLY falls back to raw `option_scores`, which would
# compute an UN-length-normalised margin for Qwen while the OLMo side used the
# normalised one -- i.e. a silently invalid cross-family comparison.
# Fix: run `scripts/enrich_per_example_normscores.py` after each merge.
# That is safe across families because `norm_lens` is a pure function of the
# dataset (the `load_task_examples` body and the
# `item_id = shard_index + ei * num_shards` convention are identical between the
# OLMo harness, the Qwen harness, and the enrich script), never of the model.
#
# PROTOCOL: chat_template=False, --add_bos 0, bs16 (matched to the OLMo bs16
# ladder this will be compared against), 8 shards/8 GPUs, 8/8-shard assert AND
# per-task n_scored assert before any merge is trusted.
#
# USAGE:  bash scripts/_run_b04_qwen_xfamily_21.sh
# Idempotent: a rung whose summary.json exists is skipped, never overwritten.
set -u

ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/Qwen--Qwen3-8b
TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
BS=16
RESULTS_ROOT=qwen3_probe2_downstream_results
PROGRESS=logs/b04_qwen_xfamily_progress.log

# Datasets are cached on wzc1; stay offline so we do not compete with the
# concurrent CAST Dolmino download for bandwidth.
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4   # .21 has 256 cores; leave plenty for the downloader

mkdir -p logs "$RESULTS_ROOT" "$HF_DATASETS_CACHE"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

assert_8shards () {
  local D="$RESULTS_ROOT/$1" MISS=0
  for g in 0 1 2 3 4 5 6 7; do
    [ -f "$D/shard${g}of8.json" ] || { log "  [SHARD MISSING] $D/shard${g}of8.json"; MISS=$((MISS+1)); }
  done
  [ $MISS -eq 0 ] && return 0
  log "  [ABORT] $MISS/8 shards missing for $1 -- NOT merging"
  return 1
}

assert_nscored () {
  local NAME="$1" SUMMARY="$RESULTS_ROOT/$1/summary.json"
  [ -f "$SUMMARY" ] || { log "  [ASSERT_N] summary.json not found for $NAME"; return 1; }
  $PY - "$SUMMARY" "$NAME" <<'PYEOF' 2>&1 | tee -a "$PROGRESS"
import json, sys
summary, name = sys.argv[1], sys.argv[2]
s = json.load(open(summary))
tasks = s.get("tasks", s)
expected = {"hellaswag":10042,"arc_challenge":1172,"arc_easy":2376,
            "piqa":1838,"openbookqa":500,"winogrande":1267}
fail = 0
for t, n in expected.items():
    got = tasks.get(t, {}).get("n_scored", -1)
    if got != n:
        print(f"  [N_SCORED MISMATCH] {name}/{t}: expected {n}, got {got}")
        fail += 1
if fail:
    print(f"  [ASSERT_N FAILED] {fail} tasks wrong for {name}")
    sys.exit(1)
print(f"  [ASSERT_N PASSED] all six n_scored correct for {name}")
PYEOF
}

# =====================================================================
# Rungs: "NAME|keep_front|n_fresh|ckpt"
# ckpt empty => full-36L base mode.
# Damage ordering is NOT assumed here -- core6 is measured and the analysis
# ranks the rungs by measured core6, exactly as the OLMo side did.
# =====================================================================
CONFIGS=(
  "qwen_base_full36_bs16|||"
  "qwen_f12k2_step200000_bs16|12|2|outputs/qwen3_minarch_armB_f12k2_200k/final.pt"
  "qwen_f12k2_step20000_bs16|12|2|outputs/qwen3_minarch_armB_f12k2_20k/final.pt"
  "qwen_f12k2_step2000_bs16|12|2|outputs/qwen3_minarch_armB_f12k2/final.pt"
  "qwen_f12k4_step2000_bs16|12|4|outputs/qwen3_minarch_armB_f12k4/final.pt"
  "qwen_scratch14L_step2000_bs16|12|2|outputs/qwen3_minarch_scratch_f12k2/final.pt"
)

log "DRIVER START on $(hostname) -- B04 cross-family Qwen ladder, ${#CONFIGS[@]} rungs, bs=$BS"

log "prepare_data"
$PY scripts/eval_qwen3_probe2_downstream.py --prepare_data --tasks "$TASKS" \
  > logs/b04_qwen_prepare.log 2>&1 || log "  prepare_data returned nonzero (continuing; datasets are cached)"
tail -3 logs/b04_qwen_prepare.log | tee -a "$PROGRESS"

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME KFL NFL CKPT <<< "$row"
  log "=========================================================="
  log "RUNG $NAME  kfl=${KFL:-BASE} nfl=${NFL:-BASE} ckpt=${CKPT:-NONE}"

  if [ -f "$RESULTS_ROOT/$NAME/summary.json" ]; then
    log "  ALREADY DONE -- skipping (never overwriting)"
    continue
  fi
  if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then
    log "  SKIP -- ckpt absent: $CKPT"
    continue
  fi

  if [ -z "$CKPT" ]; then
    ARCH_ARGS=""
  else
    ARCH_ARGS="--ckpt $CKPT --keep_front_layers $KFL --n_fresh_layers $NFL"
  fi

  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g \
    $PY scripts/eval_qwen3_probe2_downstream.py \
      --base_model "$BASE" $ARCH_ARGS \
      --tasks "$TASKS" \
      --num_shards 8 --shard_index $g \
      --batch_size $BS \
      --add_bos 0 \
      --save_per_example \
      --output_name "$NAME" \
      > "logs/b04_qwen_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  log "  shards done; asserting 8/8"
  assert_8shards "$NAME" || { log "  [FATAL] merge aborted for $NAME"; continue; }

  log "  merging"
  $PY scripts/eval_qwen3_probe2_downstream.py --merge --output_name "$NAME" \
    >> "logs/b04_qwen_${NAME}_merge.log" 2>&1

  assert_nscored "$NAME" || log "  [FATAL] n_scored assertion failed for $NAME"

  # THE CORRECTNESS FIX (see header): backfill norm_lens/norm_scores so the
  # margin the analyzer computes is length-normalised, matching the OLMo side.
  # NB: enrich takes POSITIONAL results dirs, not --flags. It self-verifies that
  # argmax(norm_scores) == the harness's stored acc_norm_score for every row, and
  # is idempotent (skips files that already carry norm_lens).
  log "  enriching per-example with norm_lens/norm_scores"
  # enrich rebuilds norm_lens by re-reading the HF datasets, and only arc_easy
  # resolves under HF_DATASETS_OFFLINE=1 -- so this ONE step needs the proxy.
  # Scoped to a subshell so the eval steps stay fully offline.
  ( export http_proxy=http://hy-proxy.woa.com:3128 \
           https_proxy=http://hy-proxy.woa.com:3128 \
           no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local,mirrors.cloud.tencent.com"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
    $PY scripts/enrich_per_example_normscores.py "$RESULTS_ROOT/$NAME" ) \
    >> "logs/b04_qwen_${NAME}_enrich.log" 2>&1 \
    || log "  [WARN] enrich returned nonzero for $NAME -- CHECK before trusting margins"

  $PY - "$RESULTS_ROOT/$NAME" <<'PYEOF' 2>&1 | tee -a "$PROGRESS"
import json, os, sys
d = sys.argv[1]
p = os.path.join(d, "per_example_hellaswag.jsonl")
if not os.path.exists(p):
    print("  [ENRICH CHECK] per_example_hellaswag.jsonl absent"); sys.exit(0)
with open(p) as f:
    r = json.loads(f.readline())
has = "norm_scores" in r and bool(r.get("norm_scores"))
print(f"  [ENRICH CHECK] norm_scores present = {has}"
      + ("" if has else "  <-- margins would be UN-normalised; do NOT compare to OLMo"))
PYEOF

  log "  core6 acc_norm:"
  $PY - "$RESULTS_ROOT/$NAME/summary.json" <<'PYEOF' 2>&1 | tee -a "$PROGRESS"
import json, sys
s = json.load(open(sys.argv[1])); t = s.get("tasks", s)
ks = ["hellaswag","arc_challenge","arc_easy","piqa","openbookqa","winogrande"]
vals = [t[k].get("acc_norm", t[k].get("acc")) for k in ks]
for k, v in zip(ks, vals):
    print(f"    {k:16s} {v:.4f}")
print(f"    {'CORE6':16s} {sum(vals)/len(vals):.4f}")
PYEOF
done

log "ALL DONE -- B04 cross-family Qwen ladder"
log "Next: run proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_5rung.py adapted to"
log "      RESULTS_ROOT=$RESULTS_ROOT and the qwen_* rung names, then compare rho to the OLMo side."
