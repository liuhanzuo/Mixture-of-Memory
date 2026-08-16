#!/usr/bin/env bash
# ============================================================================
# B04 clause-5 read-out FILL — core6 downstream eval of the 6 UNEVALUATED
# heal steps of olmo2_probe2_7B_keep14fresh2_seed1234.
#
# WHY THIS EXISTS
# ---------------
# proposal/backlog/B04-eval-fragility-incubator/DECIDABILITY_FIX_20260816.md sec 4
# Part D established that B04's gate statistic phi does not evaluate AT ALL: the
# union read-out grid is {25000,50000,100000,128000,153500,175000,200000} and
# 6 of those 7 steps have NO margin-computable eval dir on EITHER physical disk.
# Only step200000 does (olmo2_downstream_results/keep14_s1234_step200000_sv181,
# median_margin 0.108500, n=17195). The CHECKPOINTS are all present and healthy
# on wzc1; only the EVALUATION is missing. This driver fills those 6 holes.
#
# PROTOCOL OF RECORD (do not "improve" any of this -- see PROTOCOL INVARIANTS)
# ---------------------------------------------------------------------------
# Replicates scripts/_run_paperB_keep14_seedvar_local.sh:112-127 (stage "(2)
# core6 downstream") byte-for-byte on the invocation, varying ONLY --ckpt and
# --output_name. That stage is what produced the one read-out point in hand.
#
#   $PY scripts/eval_olmo2_probe2_downstream.py \
#       --base_model ../models/OLMo-2-1124-7B --ckpt <CKPT> \
#       --tasks hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa \
#       --num_shards 8 --shard_index $g --batch_size 8 \
#       --save_per_example --output_name <NAME>
#   with env CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g
#
# PROTOCOL INVARIANTS -- must be byte-identical across all 7 rungs or the
# read-out is NOT same-harness and phi is not a measurement:
#   * harness            scripts/eval_olmo2_probe2_downstream.py @ git a163a89
#                        (the commit that added norm_lens/norm_scores; a change
#                        here is a DRIVER BOUNDARY -- status/PAPERB_WITHIN_DISK_FLOOR_V3.md
#                        line 38 traced the only non-zero within-disk flip count
#                        to exactly such a boundary)
#   * base_model         ../models/OLMo-2-1124-7B
#   * tasks + order      the 6 core tasks above
#   * num_shards         8   (examples strided [shard_index::8] per task)
#   * batch_size         8
#   * max_len            1024 (harness default; NOT passed, as in the record)
#   * add_bos            0    (harness default; NOT passed, as in the record)
#   * keep/fresh         NOT passed -- read from the ckpt meta (14/2). The record
#                        omits them for the downstream stage (it passes them only
#                        to the PPL/MMLU stages), so passing them here would be a
#                        flag difference even though the values agree.
#   * --save_per_example REQUIRED. Without it median_margin is not computable and
#                        the run reproduces 7B_keep8_step100000: correct core6,
#                        primary metric still missing.
#   * chat_template      FALSE, and STRUCTURALLY so: grep finds no
#                        chat_template/apply_chat_template anywhere in the
#                        downstream harness. OLMo-2-1124-7B is a BASE LM.
#   * arch               sm_100 ONLY (LOCAL or .21). The comparator ladder and
#                        sigma_hat are LOCAL-only, so an H20 arm would confound
#                        the run-to-run term with a hardware term.
#
# n_scored IS ASSERTED, LOUDLY. A silent partial merge has destroyed a
# measurement in this project before (a 5/8 merge that looked complete). Every
# rung must clear ALL of: 8/8 shard files; per-task n_scored == HF cardinality;
# n_nan == 0; per_example_{task}.jsonl line count == the same cardinality;
# norm_scores AND norm_lens present. A rung that fails any of these is NOT
# complete, is not skipped on re-run, and aborts the driver.
#
# GPU BUDGET: 8 GPUs on ONE sm_100 node (LOCAL or .21). Never .73/.82/.104.
# Run --dry-run first: it validates everything and touches no GPU.
# ============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# CONFIG -- override any of these from the environment
# ---------------------------------------------------------------------------
WD="${WD:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
# LOCAL/.21 .venv has had no torch since 2026-08-04 -> conda. VERIFIED BY
# EXECUTION below, not by being copied from another script: the B12 driver
# scripts/launch_slorb_rank_sweep.sh was un-runnable because its own [ -x "$PY" ]
# guard pointed at a python that does not exist.
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
CKPT_DIR="${CKPT_DIR:-outputs/olmo2_probe2_7B_keep14fresh2_seed1234}"
RESULTS_ROOT="${RESULTS_ROOT:-olmo2_downstream_results}"
NGPU="${NGPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SUFFIX="${SUFFIX:-_b04fill}"
CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"

# The 6 absent rungs (DECIDABILITY_FIX_20260816.md sec 4 Part D). Order is the
# grid order, not the cheapest order: it makes a partial run's coverage legible.
STEPS="${STEPS:-25000 50000 100000 128000 153500 175000}"
# Optional 7th rung: re-evaluate step200000 under THIS driver. Not needed for the
# prereg's 6 arms, but it converts "the archived _sv181 point is same-harness" from
# an assumption into a measured fact (expect byte-identical core6 and
# median_margin 0.108500). Costs ~+0.80 occupancy GPU-h.
INCLUDE_200K="${INCLUDE_200K:-0}"

DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help) sed -n '2,70p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a (only --dry-run)"; exit 64 ;;
  esac
done

cd "$WD" || { echo "FATAL: cannot cd to WD=$WD"; exit 1; }

RUN_ID="b04_evalfill_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs "$RESULTS_ROOT"
MAIN_LOG="logs/${RUN_ID}_main.log"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }
die(){ echo "[$(date '+%F %T')] FATAL: $*" | tee -a "$MAIN_LOG" >&2; exit "${2:-1}"; }

# HF caches are pre-populated; offline so a stale hub call can never silently
# swap a dataset revision underneath the read-out.
export HF_DATASETS_CACHE="$WD/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Expected checkpoint byte sizes (MAIN's ls census 2026-08-16, re-verified here).
# An unexpected size means the file is not the checkpoint the gate names.
# ---------------------------------------------------------------------------
ckpt_expected_bytes(){
  case "$1" in
    25000|50000)                      echo 48724473567 ;;
    100000|128000|153500|175000)      echo 48724474298 ;;
    200000)                           echo 48724468275 ;;
    *) echo "" ;;
  esac
}

# HF cardinalities == analyzer's EXPECTED_N
# (proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py:85-87)
declare -A EXPECTED_N=(
  [hellaswag]=10042 [arc_challenge]=1172 [arc_easy]=2376
  [piqa]=1838 [winogrande]=1267 [openbookqa]=500
)
EXPECTED_POOLED=17195

# ---------------------------------------------------------------------------
# COMPLETENESS ORACLE. "Complete" means these assertions pass -- never merely
# that a directory exists. Same predicate is used to decide skip-on-resume and
# to accept a freshly finished rung, so the two can never disagree.
# rc 0 = complete, 1 = incomplete/absent, 2 = present but MALFORMED (hard stop).
# ---------------------------------------------------------------------------
check_rung(){ # $1 = result dir name, $2 = expected ckpt_step (optional)
  "$PY" - "$RESULTS_ROOT/$1" "${2:-}" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
want_step = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
EXP = {"hellaswag":10042,"arc_challenge":1172,"arc_easy":2376,
       "piqa":1838,"winogrande":1267,"openbookqa":500}
POOLED = 17195

def absent(m):
    print(f"INCOMPLETE {m}"); sys.exit(1)
def malformed(m):
    print(f"MALFORMED {m}"); sys.exit(2)

if not os.path.isdir(d):
    absent("dir does not exist")
shards = [f for f in os.listdir(d) if f.startswith("shard") and f.endswith("of8.json")]
if len(shards) != 8:
    absent(f"{len(shards)}/8 shard files")
s = os.path.join(d, "summary.json")
if not os.path.exists(s):
    absent("no summary.json (shards present but never merged)")
try:
    j = json.load(open(s))
except Exception as e:
    malformed(f"summary.json unreadable: {e}")
if j.get("n_shards") != 8:
    malformed(f"summary.n_shards={j.get('n_shards')} != 8 -- PARTIAL MERGE")
if j.get("add_bos") is not False:
    malformed(f"add_bos={j.get('add_bos')} != False -- protocol violation")
meta = j.get("meta") or {}
if meta.get("keep_front_layers") != 14 or meta.get("n_fresh_layers") != 2:
    malformed(f"keep/fresh = {meta.get('keep_front_layers')}/{meta.get('n_fresh_layers')} != 14/2")
# The dir must actually be the arm the gate names: seed1234, at the right step.
# A dir that evaluates a DIFFERENT ckpt but carries the right name is the worst
# possible failure -- it would silently feed a wrong y into phi.
ck = meta.get("ckpt") or ""
if "keep14fresh2_seed1234" not in ck:
    malformed(f"meta.ckpt={ck!r} is not a keep14fresh2_seed1234 checkpoint")
if want_step is not None and str(meta.get("ckpt_step")) != str(want_step):
    malformed(f"meta.ckpt_step={meta.get('ckpt_step')} != expected {want_step}")
tasks = j.get("tasks") or {}
if sorted(tasks) != sorted(EXP):
    malformed(f"task set {sorted(tasks)} != {sorted(EXP)}")
pooled = 0
for t, n in EXP.items():
    e = tasks[t]
    if e.get("skipped"):
        malformed(f"{t} SKIPPED")
    if e.get("n_scored") != n:
        malformed(f"{t} n_scored={e.get('n_scored')} expected {n}")
    if e.get("n_nan") != 0:
        malformed(f"{t} n_nan={e.get('n_nan')} != 0")
    p = os.path.join(d, f"per_example_{t}.jsonl")
    if not os.path.exists(p):
        absent(f"no per_example_{t}.jsonl (--save_per_example missing?)")
    rows = 0
    first = None
    with open(p) as f:
        for line in f:
            if line.strip():
                if first is None:
                    first = json.loads(line)
                rows += 1
    if rows != n:
        malformed(f"per_example_{t}.jsonl has {rows} rows, expected {n}")
    for k in ("norm_scores", "norm_lens", "gold_letter", "item_id"):
        if k not in first:
            malformed(f"per_example_{t} lacks '{k}' -- margin not computable")
    pooled += rows
if pooled != POOLED:
    malformed(f"pooled per_example rows {pooled} != {POOLED}")
mm = meta.get("ckpt_step")
print(f"COMPLETE ckpt_step={mm} pooled={pooled}")
sys.exit(0)
PYEOF
}

# assert 8/8 shard FILES landed before any merge is attempted
assert_shards(){ # $1 = dir
  local n; n=$(ls "$1"/shard*of8.json 2>/dev/null | wc -l)
  [ "$n" -eq 8 ] || die "only $n/8 shards in $1 -- refusing to merge a partial set" 4
  log "OK 8/8 shard files in $1"
}

# ===========================================================================
# SELF-CHECK. Everything below runs in BOTH modes and touches no GPU.
# ===========================================================================
log "===== B04 read-out eval-fill  run_id=$RUN_ID  dry_run=$DRY_RUN ====="
log "WD=$WD"
log "PY=$PY"

FAIL=0
note_fail(){ echo "  [FAIL] $*" | tee -a "$MAIN_LOG"; FAIL=1; }
note_ok(){   echo "  [ ok ] $*" | tee -a "$MAIN_LOG"; }

log "--- 1. interpreter (tested by execution, not assumed) ---"
if [ ! -x "$PY" ]; then
  note_fail "PY=$PY is not an executable file"
else
  PYVER=$("$PY" - <<'PYEOF' 2>&1
import sys
try:
    import torch, transformers
except Exception as e:
    print(f"IMPORTFAIL {type(e).__name__}: {e}"); raise SystemExit(1)
print(f"py {sys.version.split()[0]} torch {torch.__version__} "
      f"transformers {transformers.__version__} "
      f"arch_list={'sm_100' in torch.cuda.get_arch_list()}")
PYEOF
)
  if [ $? -ne 0 ]; then note_fail "PY cannot import torch/transformers: $PYVER"
  else note_ok "$PY -> $PYVER"; fi
fi

log "--- 2. harness + analyzer files ---"
for f in scripts/eval_olmo2_probe2_downstream.py scripts/eval_olmo2_probe2_ppl.py \
         proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py; do
  [ -f "$f" ] && note_ok "$f" || note_fail "missing $f"
done
HARNESS_COMMIT=$(git log -1 --format='%h %ad' --date=short \
                 -- scripts/eval_olmo2_probe2_downstream.py 2>/dev/null)
HARNESS_DIRTY=$(git status --porcelain scripts/eval_olmo2_probe2_downstream.py 2>/dev/null)
note_ok "harness last commit: ${HARNESS_COMMIT:-<no git>}"
if [ -n "$HARNESS_DIRTY" ]; then
  note_fail "harness has UNCOMMITTED changes -> would cross a driver boundary vs the archived step200000 rung: $HARNESS_DIRTY"
else
  note_ok "harness is clean vs git (no driver boundary)"
fi

log "--- 3. base model ---"
if [ -f "$BASE/config.json" ]; then
  NL=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['num_hidden_layers'])" "$BASE/config.json" 2>/dev/null)
  note_ok "$BASE (num_hidden_layers=$NL)"
else
  note_fail "missing $BASE/config.json"
fi

log "--- 4. HF dataset caches (offline mode is ON) ---"
for c in Rowan___hellaswag allenai___ai2_arc ybisk___piqa allenai___winogrande allenai___openbookqa; do
  [ -d "$HF_DATASETS_CACHE/$c" ] && note_ok "cache $c" || note_fail "missing HF cache $HF_DATASETS_CACHE/$c"
done

log "--- 5. checkpoints: existence + EXPECTED BYTE SIZE ---"
ALL_STEPS="$STEPS"
[ "$INCLUDE_200K" = "1" ] && ALL_STEPS="$ALL_STEPS 200000"
for st in $ALL_STEPS; do
  ck="$CKPT_DIR/step${st}.pt"
  want=$(ckpt_expected_bytes "$st")
  if [ ! -f "$ck" ]; then note_fail "missing ckpt $ck"; continue; fi
  got=$(stat -c %s "$ck")
  if [ -z "$want" ]; then
    note_ok "step$st size $got (no expected size registered for this step)"
  elif [ "$got" = "$want" ]; then
    note_ok "step$st $ck size $got == expected"
  else
    note_fail "step$st $ck size $got != expected $want -- NOT the checkpoint the gate names"
  fi
done

log "--- 6. output dirs: writability + per-rung completeness ---"
if [ -w "$RESULTS_ROOT" ]; then note_ok "$RESULTS_ROOT is writable"
else note_fail "$RESULTS_ROOT not writable"; fi
if [ -w logs ]; then note_ok "logs/ is writable"; else note_fail "logs/ not writable"; fi

TODO=""; SKIP=""
for st in $ALL_STEPS; do
  NAME="keep14_s1234_step${st}${SUFFIX}"
  OUT=$(check_rung "$NAME" "$st"); rc=$?
  case $rc in
    0) note_ok  "step$st -> $NAME ALREADY COMPLETE ($OUT) -- will SKIP"; SKIP="$SKIP $st" ;;
    1) note_ok  "step$st -> $NAME to be evaluated ($OUT)"; TODO="$TODO $st" ;;
    2) note_fail "step$st -> $NAME EXISTS BUT MALFORMED ($OUT). Move it aside or change SUFFIX; refusing to reuse or clobber." ;;
    *) note_fail "step$st -> completeness oracle crashed rc=$rc: $OUT" ;;
  esac
done

log "--- 7. reference rung (the protocol of record) ---"
REF=keep14_s1234_step200000_sv181
OUT=$(check_rung "$REF" 200000); rc=$?
[ $rc -eq 0 ] && note_ok "$REF $OUT (same oracle passes on the archived rung)" \
              || note_fail "$REF fails the oracle rc=$rc: $OUT -- the oracle or the record is wrong"

log "--- 8. GPUs (read-only query; no CUDA context created) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
                      --format=csv,noheader 2>/dev/null)
  if [ "${#GPUS[@]}" -lt "$NGPU" ]; then
    note_fail "only ${#GPUS[@]} GPUs visible, need $NGPU"
  else
    note_ok "${#GPUS[@]} GPUs visible (need $NGPU)"
    BUSY=0
    for g in "${GPUS[@]}"; do
      u=$(echo "$g" | awk -F', ' '{gsub(/ MiB/,"",$3); print $3+0}')
      [ "$u" -gt 2000 ] && BUSY=$((BUSY+1))
    done
    CAP=$(echo "${GPUS[0]}" | awk -F', ' '{print $2}')
    note_ok "device name string: $CAP  (NOTE: 'L20A' on LOCAL/.21 is a name-string"
    note_ok "  display bug; real hardware is B200/sm_100. Judge by capability, not name.)"
    if [ "$BUSY" -gt 0 ]; then
      echo "  [WARN] $BUSY/${#GPUS[@]} GPUs currently have >2 GiB allocated." | tee -a "$MAIN_LOG"
      echo "         This driver needs all $NGPU idle. Do NOT launch on top of a live run." | tee -a "$MAIN_LOG"
      [ "$DRY_RUN" = "0" ] && die "GPUs busy ($BUSY/${#GPUS[@]}); refusing to launch on top of another job" 5
    else
      note_ok "all GPUs idle"
    fi
  fi
  # sm_100 gate: this read-out must stay on the comparator's arch.
  # Queried via nvidia-smi's own compute_cap, NOT via torch: torch.cuda.* would
  # create a CUDA context on a GPU that may be mid-training. This stays read-only.
  SMCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$SMCAP" ]; then
    if [ "$SMCAP" = "10.0" ]; then
      note_ok "compute_cap $SMCAP = sm_100 (Blackwell/B200) -> OK"
    else
      note_fail "compute_cap $SMCAP is NOT sm_100 (need 10.0). The comparator ladder and sigma_hat are sm_100/wzc1 only; a non-sm_100 arm confounds the run-to-run term with a hardware term and makes even a FAIL uninterpretable. Use LOCAL or .21."
    fi
  else
    note_fail "could not read compute_cap from nvidia-smi -- sm_100 UNVERIFIED, refusing to assume"
  fi
else
  note_fail "nvidia-smi not on PATH"
fi

log "--- 9. host resources ---"
AVAILG=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
# 8 concurrent procs each hold an fp32 4.06 B-param model on the host (~16.2 GiB)
# before .to(device): ~130 GiB of anonymous RSS, plus page cache for a 48.7 GB ckpt.
NEEDG=180
if [ "$AVAILG" -ge "$NEEDG" ]; then note_ok "MemAvailable ${AVAILG} GiB (need ~${NEEDG} GiB for 8x fp32 4.06B + page cache)"
else note_fail "MemAvailable ${AVAILG} GiB < ~${NEEDG} GiB needed by $NGPU concurrent fp32 loads"; fi

log "--- 10. PLAN ---"
NTODO=$(echo $TODO | wc -w); NSKIP=$(echo $SKIP | wc -w)
log "  protocol : eval_olmo2_probe2_downstream.py, 8 shards, batch_size $BATCH_SIZE,"
log "             --save_per_example, add_bos 0 (default), max_len 1024 (default),"
log "             chat_template absent by construction, keep/fresh from ckpt meta"
log "  tasks    : $CORE_TASKS"
log "  expect   : per-task n_scored $(for t in hellaswag arc_challenge arc_easy piqa winogrande openbookqa; do printf '%s=%s ' "$t" "${EXPECTED_N[$t]}"; done)"
log "             pooled $EXPECTED_POOLED, n_nan 0, 8/8 shards"
log "  to run   :${TODO:- (none)}   ($NTODO rungs)"
log "  skipping :${SKIP:- (none)}   ($NSKIP already complete)"
log "  outputs  : $RESULTS_ROOT/keep14_s1234_step<STEP>${SUFFIX}/"
for st in $TODO; do
  log "    step$st: $CKPT_DIR/step${st}.pt -> $RESULTS_ROOT/keep14_s1234_step${st}${SUFFIX}"
done
# Cost, from measured anchors (see EVAL_FILL_READY_20260816.md sec 4):
#   cold 48.7 GB ckpt load, 8 concurrent procs  ~325 s   (sv181 PPL stage, COLD)
#   core6 scoring after load                     ~33 s   (sv181 core stage)
#   merge                                         ~4 s
PER_RUNG_S=362
log "  cost     : ~${PER_RUNG_S}s/rung wall = cold ckpt read ~325s + scoring ~33s + merge ~4s"
log "             MEASURED anchors (EVAL_FILL_READY_20260816.md sec 4), NOT copied:"
log "               COLD load  logs/sv181_main.log:2->  + sv181_ppl_*_shard*.log = 338-348s"
log "               WARM load  logs/sv181_main.log:5-6 stage 121s, of which 89s load + 30s score"
log "               The artifact's 121s/rung anchor is the WARM stage: the PPL stage"
log "               had already read that same ckpt minutes earlier. Each of the 6 new"
log "               rungs is a FIRST touch, so 121s understates it ~3x."
log "             $NTODO rungs -> ~$((NTODO*PER_RUNG_S/60)) min wall, ~$(awk "BEGIN{printf \"%.2f\", $NTODO*$PER_RUNG_S*$NGPU/3600}") occupancy GPU-h (8 cards held)"
log "             compute-only (scoring, excluding the IO-bound read): ~$(awk "BEGIN{printf \"%.2f\", $NTODO*33*$NGPU/3600}") GPU-h"
log "             set PREFETCH=1 to warm the page cache per rung on CPU first; that"
log "             moves the 325s off the GPU clock (~$(awk "BEGIN{printf \"%.2f\", $NTODO*(89+33+4)*$NGPU/3600}") GPU-h) but does not change the total wall."

if [ "$FAIL" -ne 0 ]; then
  die "self-check FAILED (see [FAIL] lines above). Nothing launched, no GPU touched." 2
fi
log "SELF-CHECK PASSED ($NTODO to run, $NSKIP complete)"

if [ "$DRY_RUN" = "1" ]; then
  log "--dry-run: validation only. No GPU touched, nothing written to $RESULTS_ROOT/."
  exit 0
fi

if [ "$NTODO" -eq 0 ]; then
  log "nothing to do -- all requested rungs already complete. Exiting 0."
  exit 0
fi

# ===========================================================================
# REAL RUN
# ===========================================================================
log "cache datasets once (offline, no network, ~1s when warm)"
"$PY" scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS" \
  > "logs/${RUN_ID}_prep_core.log" 2>&1 \
  || die "--prepare_data failed; see logs/${RUN_ID}_prep_core.log" 6

for st in $TODO; do
  CKPT="$CKPT_DIR/step${st}.pt"
  NAME="keep14_s1234_step${st}${SUFFIX}"
  log "################ RUNG step=$st ckpt=$CKPT -> $NAME ################"
  # re-assert size at launch time: the dry-run may have been hours earlier
  want=$(ckpt_expected_bytes "$st"); got=$(stat -c %s "$CKPT" 2>/dev/null)
  [ -n "$got" ] || die "ckpt vanished: $CKPT" 3
  if [ -n "$want" ] && [ "$got" != "$want" ]; then
    die "ckpt $CKPT size $got != expected $want" 3
  fi

  T0=$(date +%s)
  # OPTIONAL page-cache warm-up. Purely an IO reordering: it reads the ckpt bytes
  # into the host page cache before the 8 GPU workers start, so their torch.load
  # hits cache (measured 89s) instead of cold CephFS (measured 338-348s). It cannot
  # change any number the eval produces -- it never touches the model or the data
  # pipeline. Off by default so the default path stays closest to the record.
  if [ "${PREFETCH:-0}" = "1" ]; then
    log "  prefetch: warming page cache for $CKPT (8 parallel streams, CPU only)"
    P0=$(date +%s)
    CHUNK=$(( (got / NGPU / 8388608) + 1 ))
    for i in $(seq 0 $((NGPU-1))); do
      dd if="$CKPT" of=/dev/null bs=8M count="$CHUNK" skip=$((i*CHUNK)) 2>/dev/null &
    done
    wait
    log "  prefetch: done in $(( $(date +%s) - P0 ))s"
  fi
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g "$PY" scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE_TASKS" \
      --num_shards 8 --shard_index $g --batch_size "$BATCH_SIZE" \
      --save_per_example --output_name "$NAME" \
      > "logs/${RUN_ID}_core_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_shards "$RESULTS_ROOT/$NAME"
  "$PY" scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" \
    --results_root "$RESULTS_ROOT" 2>&1 | tee -a "$MAIN_LOG" \
    || die "merge failed for $NAME" 7

  # The gate's own acceptance test, applied to the rung we just made. A rung that
  # fails here is NOT recorded as done and the driver stops -- a plausible-looking
  # short merge must never reach the analyzer.
  OUT=$(check_rung "$NAME" "$st"); rc=$?
  [ $rc -eq 0 ] || die "post-merge completeness check FAILED for $NAME (rc=$rc): $OUT" 8
  T1=$(date +%s)
  log "RUNG step=$st DONE in $((T1-T0))s -- $OUT"
done

log "--- final re-verification of every requested rung ---"
BAD=0
for st in $ALL_STEPS; do
  NAME="keep14_s1234_step${st}${SUFFIX}"
  OUT=$(check_rung "$NAME" "$st"); rc=$?
  if [ $rc -eq 0 ]; then log "  step$st OK  $OUT"; else log "  step$st BAD rc=$rc $OUT"; BAD=1; fi
done
[ "$BAD" -eq 0 ] || die "at least one rung is not complete -- do NOT feed this read-out to the gate" 9

log "===== ALL RUNGS COMPLETE ====="
log "The read-out is now margin-computable at: $ALL_STEPS (+ 200000 via $REF)."
log "NEXT (0 GPU): revision 3's read-out path IS now implemented in the analyzer"
log "      (clause5_revision3 / GRID_I / two-grid combine, added 2026-08-16 -- see"
log "      proposal/backlog/B04-eval-fragility-incubator/READOUT_PATH_20260816.md)."
log "      Compute the gate with:"
log "        python3 proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py --readout-only"
log "      It exits 0 only on a computed verdict; 3 = READOUT_ABSENT, 4 = PROTOCOL_VIOLATION"
log "      / FIELD_ASYMMETRY, 5 = DENOMINATOR_UNRESOLVED / FLOOR_UNMEASURABLE. Check \$?."
log "      NOTE the dir-name coupling: the analyzer identifies arms by"
log "      summary.json.meta.ckpt + .ckpt_step, NOT by dir name, so SUFFIX=$SUFFIX is free."
