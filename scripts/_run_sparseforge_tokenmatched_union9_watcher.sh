#!/usr/bin/env bash
# ============================================================================
# Task #246 CLOSEOUT — chained watcher: wait for the token-matched SparseForge
# arm to finish, then score it OFFLINE on the union-9 harness.
#
#   ARM=noslorb bash scripts/_run_sparseforge_tokenmatched_union9_watcher.sh   # run ON .21
#   ARM=slorb   bash scripts/_run_sparseforge_tokenmatched_union9_watcher.sh   # run ON .21
#
# BOTH watchers run on .21 -- it is the only node with the pinned union-9 harness
# stack (lm_eval 0.4.8 + transformers 4.57.6). See the "SCORED ON .21" block below.
# They take disjoint GPU halves (noslorb 0-3, slorb 4-7) so they can run at once.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS — the protocol gap
# ---------------------------------------------------------------------------
# scripts/_run_sparseforge_tokenmatched.sh:294 configures the IN-RUN eval as
#
#   --lm_eval_tasks "hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,race"
#
# That is SEVEN tasks. The union-9 table's protocol is NINE
# (scripts/_sparseforge_same_harness_21.sh:53):
#
#   TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa
#
# boolq and rte are absent. Placing a 7-task mean beside the other arms' 9-task
# mean would be a category error, and aggregate_zeroshot_union9.py refuses it
# (line 103: `missing task results: [...] -- this arm's row is INVALID`).
#
# ---------------------------------------------------------------------------
# ⚠️ BUT THE GAP IS BIGGER THAN "TWO MISSING TASKS" — VERIFIED 2026-08-13
# ---------------------------------------------------------------------------
# These two runs will produce **NO in-run lm_eval numbers at all**, not even the
# seven. `--finalize_lm_eval True` is a dead flag here. Proof, from main_llama.py:
#
#   L2102  if iter_num % args.eval_interval == 0:            # eval block
#   L2248      if finalization_done and args.finalize_lm_eval:   # <- lm_eval gate
#   L3052  if iter_num > max_iters:
#   L3053      if not finalization_done:
#   L3215          finalization_done = True                  # set only HERE
#   L3436          extra = int(args.final_finetune_iters)     # == 0 for both arms
#   L3437          if extra > 0:  ... L3466 continue
#   L3467          else:          ... L3470 break             # <- leaves `while True:`
#
# `finalization_done` flips to True only after `iter_num > max_iters`, and with
# `final_finetune_iters = 0` (set deliberately at _run_sparseforge_tokenmatched.sh's
# FINAL_FT=0, to avoid re-importing the qa_format_sft_llama contamination) the
# very next thing the loop does is `break`. Control never returns to L2102, so
# the L2248 gate is never evaluated. Confirmed empirically: the published
# 17000-iter run has final_finetune_iters=3000 and DOES have best_lm_eval.json;
# every `final_finetune_iters=0` Llama-2 run in out_llama/ has none.
#
# => This script is not "topping up boolq+rte". It is the ONLY source of
#    zero-shot numbers for these two arms. Everything must be measured here.
#
# ---------------------------------------------------------------------------
# EXPORT VARIANT — never `--slorb drop` on an SLoRB-TRAINED checkpoint
# ---------------------------------------------------------------------------
# `--slorb drop` DISCARDS the trained low-rank branch. On a model that trained
# WITH it, that is a post-hoc AMPUTATION, and its cost (~4.9 pp AST-7) is a
# statement about amputation damage, not about the method. This is Defect 1 of
# baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md's CORRECTION block, which
# retracted a headline conclusion built on exactly that confusion. The whole
# point of these two arms is that ±SLoRB is decided AT TRAINING TIME.
#
#   ARM=slorb    ckpt HAS SLoRB_Weight/x_proj  -> --mask hard --slorb fold
#                (fold = exact linear algebra, keeps the branch the model learned)
#   ARM=noslorb  ckpt has NO SLoRB tensors     -> --mask hard --slorb drop
#
# For `noslorb`, `drop` is NOT an amputation: verified by probing the ckpt
# pickle, `SLoRB_Weight` and `x_proj` are absent, so `drop` is a strict no-op on
# a model that never had the branch. It is also FORCED: `--slorb fold` on that
# ckpt hard-exits at export_sparseforge_to_hf.py:181
# (`--slorb fold requested but ...SLoRB_Weight/...x_proj missing`).
# Hence the mapping is arm-conditional and asserted below, never hand-picked.
#
# ---------------------------------------------------------------------------
# 2:4 COLUMN ELIGIBILITY — two different kinds of verify FAIL
# ---------------------------------------------------------------------------
# verify_2of4_hf_export.py gates on `tiles_gt2 == 0 && zero_frac >= 0.5-1e-4 &&
# len(scope) == 224` (its lines 118-130). Two FAIL modes must not be conflated:
#
#   EXPECTED FAIL — an SLoRB-folded export. The fold writes into positions the
#     2:4 mask pruned, so zero_frac collapses to ~0 and exact_2of4_frac == 0.
#     This is by design. Such an arm is scored and reported, but is BARRED from
#     any 2:4 column. Script treats this as `2of4_eligible=false`, not an error.
#   REAL FAIL — an export that was supposed to be 2:4 (noslorb/hard_drop) and
#     is not. That is a bug or a corrupt ckpt. Script ABORTS without scoring.
#
# ---------------------------------------------------------------------------
# PPL — measured at BOTH 2048 and 4096, each labelled
# ---------------------------------------------------------------------------
# SPEC.md:213's "the whole PPL column is 2048" assumption was falsified: the
# SparseForge headline 6.2179 is a seqlen-4096 number that sat in a 2048 column
# (commit 501dafb). So both are measured and written to separate directories
# (ppl2048/, ppl4096/), each ppl_metrics.json carrying its own "seqlen" field.
# No caller has to infer which is which.
#
# ---------------------------------------------------------------------------
# RTE INTEGRAL COUNT
# ---------------------------------------------------------------------------
# RTE n=277. Existing arms: dense 175, hard_fold/soft_fold 139, AST 184,
# CAST-repro 207, hard_drop 158, Wanda 148. The published 69.82 was caught as a
# transcription error precisely by reconstructing k = acc*277 = 139. The summary
# emits k for these arms too, and asserts it is within 0.01 of an integer.
#
# ---------------------------------------------------------------------------
# ⚠️ BOTH ARMS MUST BE SCORED ON .21 — LOCAL CANNOT PRODUCE A COMPARABLE ROW
# ---------------------------------------------------------------------------
# Measured 2026-08-13:
#   .21    /opt/conda/envs/torch-base : lm_eval 0.4.8, transformers 4.57.6  <-- the table's stack
#   LOCAL  /opt/conda/envs/torch-base : NO lm_eval at all, transformers 5.13.1
#   LOCAL  Mixture-of-Memory/.venv    : NO lm_eval, no torch
# Every existing union-9 arm records `"transformers_version": "4.57.6"` and
# `"git_hash": "b86c479"` in its results_*.json. Scoring on LOCAL would be a
# different stack, i.e. exactly the cross-harness error that already cost this
# project a retraction (-0.346 pp AST-7 offset on the AST arm). The preflight
# therefore hard-refuses rather than silently producing a non-comparable number.
#
# Consequence: the `noslorb` arm TRAINS on LOCAL but is SCORED on .21. Both live
# on the same wzc1 disk, so no copy is needed -- verified that .21 can stat
# LOCAL's out_llama_tokenmatched_noslorb/ and read its log.
#
# ---------------------------------------------------------------------------
# GPU DISCIPLINE
# ---------------------------------------------------------------------------
# This watcher sleeps until (a) the arm's trainer is gone, (b) the checkpoint
# passes a completeness guard, and (c) its OWN 4 assigned GPUs are free.
#
# Liveness is judged by PROCESS EXISTENCE (local) or LOG-MTIME PROGRESS (remote
# arm), never by instantaneous `nvidia-smi` utilisation — a 0% sample can just be
# a model-load gap (memory: one-sample-is-not-a-trend-or-state). For the arm that
# trains on another node the watcher cannot pgrep it, so it uses the shared log:
# an mtime younger than LOG_STALE_S means the remote trainer is still writing.
#
# GPU SPLIT mirrors _sparseforge_same_harness_21.sh's run_one() topology exactly
# (0,1,2,3 and 4,5,6,7 -- and every arm already in the table was scored on 4 GPUs
# with parallelize=True, per its lm_eval.log "max memory per GPU" line). Giving
# each arm a disjoint half lets both watchers run concurrently on .21 without
# contending, and keeps the per-arm GPU count identical to the existing rows.
#
# This script never kills the trainer. It only waits. `kill -9 <PID>` on the
# watcher's own PID is the way to stop it; `pkill -f` is banned repo-wide
# (memory: kill-remote-gpu-job-by-pid-not-pkill).
#
# ---------------------------------------------------------------------------
# WHAT THIS TABLE CAN AND CANNOT ANSWER
# ---------------------------------------------------------------------------
# CAN:    whether training-time SLoRB helps, at matched corpus/tokens/harness.
# CANNOT: whether SparseForge's mask search is good. Both arms train on
#         dolmino-mix-1124-llama2, not SparseForge's own qa_format_sft_llama.
# Full statement: baselines/cast_repro/SPARSEFORGE_TOKENMATCHED_UNION9_PLAN.md
# ============================================================================
set -u

ARM="${ARM:-}"
# ⚠️ TRAIN_NODE / TRAIN_LOG are OVERRIDABLE (`${VAR:-default}`), and must stay so.
# They were originally hard-assigned (`TRAIN_NODE=local`), which silently ignored
# any caller override. After the 2026-08-13 restart the topology INVERTED —
# noslorb now trains on the scoring box (LOCAL, which is itself 28.89.19.21) while
# slorb trains on .212 — so the baked-in mapping became exactly backwards. With a
# hard assignment, ARM=slorb would pgrep locally for a trainer living on another
# host, find nothing, conclude "training finished", and score a MID-TRAINING
# checkpoint hours early. The defaults below are kept as the historical values;
# scripts/_rearm_sparseforge_union9_watchers.sh passes the current topology in.
case "$ARM" in
  slorb)
    SLORB_VARIANT=fold; EXPECT_SLORB_TENSORS=1
    GPUS="${GPUS:-4,5,6,7}"
    TRAIN_NODE="${TRAIN_NODE:-local}"
    TRAIN_LOG="${TRAIN_LOG:-logs/sparseforge_tokenmatched_slorb_0811_225720.log}"
    ;;
  noslorb)
    SLORB_VARIANT=drop; EXPECT_SLORB_TENSORS=0
    GPUS="${GPUS:-0,1,2,3}"
    TRAIN_NODE="${TRAIN_NODE:-remote}"
    TRAIN_LOG="${TRAIN_LOG:-logs/sparseforge_tokenmatched_noslorb_0812_022335.log}"
    ;;
  *) echo "FATAL: set ARM=slorb or ARM=noslorb"; exit 2 ;;
esac
VARIANT="hard_${SLORB_VARIANT}"
GPU0="${GPUS%%,*}"       # first of my assigned GPUs, for the single-GPU stages

ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
MOM="$ROOT/Mixture-of-Memory"
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
LM_EVAL="${LM_EVAL_BIN:-/opt/conda/envs/torch-base/bin/lm_eval}"

TOOLS=$MOM/baselines/cast_repro/tools
EXPORT=$TOOLS/export_sparseforge_to_hf.py
VERIFY=$TOOLS/verify_2of4_hf_export.py
AGG9=$TOOLS/aggregate_zeroshot_union9.py
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
REF_MODEL=$ROOT/models/Llama--Llama2-7b

TRAIN_OUT_DIR="$ROOT/out_llama_tokenmatched_$ARM"
OUT=$ROOT/outputs/cast_eval_spec/sparseforge_tokenmatched_$ARM
EXPDIR=$ROOT/outputs/sparseforge_tokenmatched_${ARM}_hf
PROG=$MOM/logs/sparseforge_tm_union9_${ARM}_progress.log

# Byte-identical to _sparseforge_same_harness_21.sh:53 / _missing_baselines_union9_21.sh:58.
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

# boolq + rte resolve through hub parquet redirects, so the proxy is REQUIRED.
export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'

# Watcher tunables.
STALE_S="${STALE_S:-300}"          # ckpt mtime must be this old before trusting it
SIZE_SETTLE_S="${SIZE_SETTLE_S:-60}"
POLL_S="${POLL_S:-300}"
MAX_WAIT_H="${MAX_WAIT_H:-48}"
GPU_FREE_MIB="${GPU_FREE_MIB:-8000}"
LOG_STALE_S="${LOG_STALE_S:-1800}"   # remote-arm liveness: training log idle this long = done

mkdir -p "$MOM/logs" "$OUT" "$EXPDIR"

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

note "=============================================================="
note "ARM=$ARM  variant=$VARIANT (mask=hard slorb=$SLORB_VARIANT)"
note "train_out=$TRAIN_OUT_DIR"
note "train_node=$TRAIN_NODE  train_log=$TRAIN_LOG"
note "gpus=$GPUS"
note "out=$OUT"
note "expdir=$EXPDIR"
note "tasks=$TASKS"
note "=============================================================="

# ------------------------------------------------------------------ PREFLIGHT
# Fail-closed on missing tooling: better to die now than 13 h from now.
for f in "$EXPORT" "$VERIFY" "$AGG9" "$HARNESS_PPL" "$WIKI" \
         "$REF_MODEL/model.safetensors.index.json" "$REF_MODEL/tokenizer.model"; do
  [ -e "$f" ] || { note "FATAL missing asset: $f"; exit 3; }
done
[ -x "$PY" ] || { note "FATAL python not executable: $PY"; exit 3; }
if [ ! -x "$LM_EVAL" ]; then
  note "FATAL lm_eval binary not found at $LM_EVAL"
  note "  NOTE: LOCAL's /opt/conda/envs/torch-base has NO lm_eval and transformers 5.13.1,"
  note "        while every existing union-9 arm was scored with lm_eval 0.4.8 +"
  note "        transformers 4.57.6 (results_*.json 'transformers_version')."
  note "        Scoring here would be a DIFFERENT harness. Run this arm's eval on .21,"
  note "        or install the pinned stack first. Set LM_EVAL_BIN to override."
  exit 3
fi
note "preflight: tooling OK ($LM_EVAL)"

# Harness-identity assertion. A silent stack drift is exactly the class of error
# that produced the retracted cross-harness comparison (-0.346 pp AST-7 offset).
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import importlib.metadata as m
want = {"lm_eval": "0.4.8", "transformers": "4.57.6"}
bad = []
for pkg, exp in want.items():
    try:
        got = m.version(pkg)
    except Exception as e:
        got = f"<missing: {e.__class__.__name__}>"
    tag = "OK " if got == exp else "DRIFT"
    print(f"[harness] {tag} {pkg}: got {got} expected {exp}")
    if got != exp:
        bad.append(pkg)
print("[harness] VERDICT: " + ("MATCH" if not bad else f"MISMATCH on {bad}"))
raise SystemExit(0 if not bad else 21)
PYEOF
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  note "FATAL harness stack does not match the union-9 arms (lm_eval 0.4.8 / transformers 4.57.6)."
  note "      Refusing to add a row measured on a different stack. Fix the env or use .21."
  exit 21
fi

# Proxy reachability -- boolq/rte need the hub.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
       https://huggingface.co/api/datasets/aps/super_glue 2>/dev/null || echo 000)
if [ "$code" != "200" ]; then
  note "WARN proxy check returned HTTP $code for aps/super_glue."
  note "     boolq/rte may fail to resolve. Continuing (cache may cover it), but watch STAGE 4."
else
  note "preflight: proxy OK (hub HTTP 200)"
fi

# ------------------------------------------------------------------ CKPT DISCOVERY
# The trainer writes into $TRAIN_OUT_DIR/models_<...>_<timestamp>/ and maintains a
# `last` symlink to it.
#
# ⚠️ That symlink is stored RELATIVE TO $ROOT (the trainer's CWD), not relative to
# its own directory: `last -> out_llama_tokenmatched_slorb/models_...`. So plain
# `readlink -f` yields EMPTY (it looks for
# .../out_llama_tokenmatched_slorb/out_llama_tokenmatched_slorb/models_...).
# Verified 2026-08-13 on both arms -- an earlier version of this function silently
# reported "no run dir" because of it. Resolve the link text against $ROOT, and
# fall back to the newest non-DIVERGED models_* directory.
#
# DIVERGED_* dirs (the abandoned lr=1e-4 attempt, which blew up to ppl 2230 at
# iter ~860) must never be selected.
resolve_run_dir() {
  local link d best="" bestt=0 t
  if [ -L "$TRAIN_OUT_DIR/last" ]; then
    link=$(readlink "$TRAIN_OUT_DIR/last" 2>/dev/null || true)
    if [ -n "$link" ]; then
      case "$link" in
        /*) d="$link" ;;
        *)  d="$ROOT/$link" ;;         # link text is relative to $ROOT, not to us
      esac
      case "$(basename "$d")" in DIVERGED_*) d="" ;; esac
      if [ -n "$d" ] && [ -d "$d" ]; then echo "$d"; return 0; fi
    fi
  fi
  for d in "$TRAIN_OUT_DIR"/models_*; do
    [ -d "$d" ] || continue
    case "$(basename "$d")" in DIVERGED_*) continue ;; esac
    t=$(stat -c %Y "$d" 2>/dev/null) || continue
    if [ "$t" -gt "$bestt" ]; then bestt=$t; best=$d; fi
  done
  [ -n "$best" ] && { echo "$best"; return 0; }
  return 1
}

# Trainer processes for THIS arm only. The arm name appears in the cmdline via
# `--out_dir out_llama_tokenmatched_<arm>` (verified against /proc/<pid>/cmdline),
# so the two arms never match each other.
#
# ⚠️ The exclusion filter must key on the SCRIPT NAME, not on the substring
# "lm_eval". The trainer's own cmdline contains `--lm_eval_tasks ...` and
# `--lm_eval_batch_size 64`, so a `*lm_eval*` glob filters out every genuine
# trainer PID -- verified on .21, where it silently discarded all 9 of them and
# made the watcher believe training had finished. Match `*/eval_*.py` or
# ` lm_eval ` as an invoked program instead. (Related:
# memory kill-hung-train-must-exclude-eval-procs -- same trap, opposite polarity.)
trainer_pids() {
  pgrep -f "main_llama.py.*out_llama_tokenmatched_${ARM}\b" 2>/dev/null \
    | while read -r p; do
        c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in
          *"bin/lm_eval "*|*"eval_hf_sparse_model.py"*|*"verify_2of4_hf_export.py"*) continue ;;
        esac
        case "$c" in *main_llama.py*) echo "$p" ;; esac
      done
}

# Is the arm's trainer still working? Two regimes:
#  - TRAIN_NODE=local : authoritative -- pgrep the process on this node.
#  - TRAIN_NODE=remote: the trainer runs on the other wzc1 node, invisible to
#    pgrep here. Use the shared training log's mtime as the progress signal. It
#    is appended on every iteration (tqdm writes ~1 line / 45-55 s), so an mtime
#    older than LOG_STALE_S means it stopped. This is a DURATION test over a
#    30-min window, not a single instantaneous sample.
trainer_alive() {
  if [ "$TRAIN_NODE" = "local" ]; then
    local pids; pids=$(trainer_pids | tr '\n' ' ')
    if [ -n "${pids// /}" ]; then
      note "training (local) still alive (pids: $pids); waiting"
      return 0
    fi
    return 1
  fi
  # remote
  if [ ! -f "$MOM/$TRAIN_LOG" ]; then
    note "remote training log $TRAIN_LOG absent; cannot confirm completion; waiting"
    return 0
  fi
  local lage; lage=$(( $(date +%s) - $(stat -c %Y "$MOM/$TRAIN_LOG") ))
  if [ "$lage" -lt "$LOG_STALE_S" ]; then
    note "remote training log advanced ${lage}s ago (< ${LOG_STALE_S}s); still running; waiting"
    return 0
  fi
  note "remote training log idle ${lage}s (>= ${LOG_STALE_S}s); treating remote trainer as finished"
  return 1
}

# Memory held on THIS arm's assigned GPUs only, so the two watchers on .21 do not
# block each other. `nvidia-smi -i <list>` restricts the query to that subset.
gpu_used_mib() {
  nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}

# --------------------------------------------------------------- COMPLETENESS
# Same criterion as scripts/_run_a04_stageB.sh's complete(): exists -> mtime
# stale -> size stable across a settle window -> torch.load probe succeeds.
# The torch.load probe is NOT paranoia: out_llama/..._20260306_211245/model.pt is
# 6.7 GB against a 47.7 GB sibling and raises PytorchStreamReader "failed finding
# central directory" -- a truncated checkpoint that every size-only guard passes.
CK=""
complete() {
  local run_dir age s1 s2
  run_dir=$(resolve_run_dir) || { note "no run dir under $TRAIN_OUT_DIR yet"; return 1; }

  # Prefer the finalized/terminal checkpoint. With final_finetune_iters=0 the
  # trainer's last write is the FINALIZED model.pt (main_llama.py:3395), which is
  # the one carrying hard masks; model_final.pt only exists on runs that did a
  # post-finalize finetune. model_best.pt is a PPL-best mid-run snapshot and is
  # NOT the endpoint, so it is never selected.
  local cand=""
  for n in model_final.pt model.pt; do
    [ -f "$run_dir/$n" ] && { cand="$run_dir/$n"; break; }
  done
  [ -n "$cand" ] || { note "no model_final.pt/model.pt in $run_dir yet"; return 1; }

  local pids
  if trainer_alive; then
    return 1
  fi

  age=$(( $(date +%s) - $(stat -c %Y "$cand") ))
  if [ "$age" -lt "$STALE_S" ]; then
    note "$(basename "$cand") only ${age}s old (< ${STALE_S}s); waiting"
    return 1
  fi

  s1=$(stat -c %s "$cand" 2>/dev/null) || return 1
  sleep "$SIZE_SETTLE_S"
  s2=$(stat -c %s "$cand" 2>/dev/null) || return 1
  if [ "$s1" != "$s2" ]; then
    note "still growing ($s1 -> $s2); waiting"
    return 1
  fi

  local used; used=$(gpu_used_mib)
  if [ "$used" -gt "$GPU_FREE_MIB" ]; then
    note "REFUSE: ${used}MiB held on my GPUs ($GPUS) (> ${GPU_FREE_MIB}); not competing for cards"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>&1 \
      | head -12 | tee -a "$PROG"
    return 1
  fi

  # torch.load probe + structural assertions, all in one pass.
  #
  # ⚠️ The exit status is taken from PIPESTATUS[0], NOT from `if ! ... | tee`.
  # `tee` is the last stage of the pipeline, so it almost always succeeds and
  # would swallow the probe's failure. Verified: with `if ! "$PY" ... | tee`, a
  # deliberately wrong-arm checkpoint printed "PROBE FAIL" and the watcher then
  # declared the ckpt COMPLETE and proceeded to score it.
  "$PY" - "$cand" "$EXPECT_SLORB_TENSORS" <<'PYEOF' 2>&1 | tee -a "$PROG"
import sys, torch
p, expect_slorb = sys.argv[1], int(sys.argv[2])
blob = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
sd = blob.get("model_state_dict")
if sd is None:
    print("  PROBE FAIL: no 'model_state_dict' key"); raise SystemExit(1)
n_slorb = sum(1 for k in sd if k.endswith(".SLoRB_Weight"))
n_xproj = sum(1 for k in sd if k.endswith(".x_proj"))
n_mask  = sum(1 for k in sd if k.endswith(".mask"))
print(f"  torch.load OK: {len(sd)} tensors iter_num={blob.get('iter_num')} "
      f"finalization_done={blob.get('finalization_done')} "
      f"masks={n_mask} SLoRB_Weight={n_slorb} x_proj={n_xproj}")
if expect_slorb and (n_slorb == 0 or n_xproj == 0):
    print("  PROBE FAIL: ARM=slorb but ckpt has no SLoRB tensors -> wrong ckpt")
    raise SystemExit(1)
if not expect_slorb and (n_slorb or n_xproj):
    print("  PROBE FAIL: ARM=noslorb but ckpt HAS SLoRB tensors -> wrong ckpt")
    raise SystemExit(1)
if n_mask == 0:
    print("  PROBE FAIL: no .mask buffers; export cannot derive 2:4 support")
    raise SystemExit(1)
raise SystemExit(0)
PYEOF
  probe_rc=${PIPESTATUS[0]}
  if [ "$probe_rc" -ne 0 ]; then
    note "REFUSE: torch.load / structure probe FAILED (rc=$probe_rc) on $cand"
    return 1
  fi

  CK="$cand"
  note "ckpt COMPLETE: $CK (size $s1, stable, torch.load OK, GPU free ${used}MiB)"
  return 0
}

# ------------------------------------------------------------------ WAIT LOOP
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
note "entering wait loop (poll ${POLL_S}s, budget ${MAX_WAIT_H}h)"
while :; do
  if complete; then break; fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    note "FATAL wait budget ${MAX_WAIT_H}h exhausted without a complete ckpt. NOT scoring."
    exit 9
  fi
  sleep "$POLL_S"
done

ITER=$("$PY" -c "
import torch,sys
b=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True)
print(b.get('iter_num','unk'))" "$CK" 2>/dev/null || echo unk)
note "scoring ckpt=$CK iter_num=$ITER"

# ------------------------------------------------------------------ STAGE 1: export
note "=== STAGE 1: export $VARIANT (mask=hard slorb=$SLORB_VARIANT) ==="
if [ -f "$EXPDIR/$VARIANT/sparseforge_export_meta.json" ]; then
  note "--- $VARIANT already exported; reusing"
else
  "$PY" "$EXPORT" --ckpt "$CK" --output "$EXPDIR/$VARIANT" \
      --mask hard --slorb "$SLORB_VARIANT" \
      --model "$REF_MODEL" --project-root "$ROOT" 2>&1 \
    | tee "$OUT/export_$VARIANT.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    note "FATAL export failed rc=$rc -- a wrong export is worse than no number"
    exit "$rc"
  fi
fi
note "=== STAGE 1 DONE ==="

# ------------------------------------------------------------------ STAGE 2: 2:4 verify (PRE)
# Raw gate output is preserved verbatim; eligibility is decided from the rc, and
# the two FAIL modes are distinguished (see header).
note "=== STAGE 2: verify 2:4 $VARIANT (PRE-inference) ==="
CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$VERIFY" \
    --model "$EXPDIR/$VARIANT" --sample-layers 12 --seed 0 2>&1 \
  | tee "$OUT/verify_2of4_${VARIANT}_pre.log"
pre_rc=${PIPESTATUS[0]}
note "STAGE 2 rc=$pre_rc"

if [ "$SLORB_VARIANT" = "drop" ]; then
  # noslorb: `drop` is a no-op (no branch existed), so this MUST be exact 2:4.
  # A FAIL here is a REAL failure -> abort before spending GPU-hours.
  if [ "$pre_rc" -ne 0 ]; then
    note "FATAL $VARIANT failed the 2:4 gate but was expected to PASS."
    note "      ARM=noslorb trained without SLoRB, so slorb=drop removes nothing"
    note "      and the export must be exact 2:4. This is a REAL failure (bad ckpt"
    note "      or export bug), NOT the by-design fold failure. Not scoring."
    exit "$pre_rc"
  fi
  ELIGIBLE_2OF4=true
  note "$VARIANT PASSES the 2:4 gate -> eligible for the 2:4 column"
else
  # slorb: folding writes into pruned positions, so FAIL is EXPECTED, by design.
  if [ "$pre_rc" -eq 0 ]; then
    note "WARN $VARIANT PASSED the 2:4 gate. Folding a live SLoRB branch cannot leave"
    note "     the weight 2:4 unless the branch is ~a no-op. Investigate before trusting."
    ELIGIBLE_2OF4=suspicious
  else
    ELIGIBLE_2OF4=false
    note "$VARIANT FAILS the 2:4 gate -- EXPECTED (SLoRB fold densifies the weight)."
    note "     This arm will be scored and reported but is BARRED from any 2:4 column."
  fi
fi

# ------------------------------------------------------------------ STAGE 3: PPL @2048 and @4096
note "=== STAGE 3: WikiText-2 PPL at BOTH seqlens ==="
for SEQ in 4096 2048; do
  o=$OUT/$VARIANT/ppl${SEQ}
  mkdir -p "$o"
  note "--- ppl@${SEQ} $VARIANT"
  CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$HARNESS_PPL" \
      --model "$EXPDIR/$VARIANT" \
      --output_dir "$o" \
      --wiki_text "$WIKI" \
      --seqlen "$SEQ" \
      --wiki_tokens 100000000 \
      --device cuda:0 2>&1 | tee "$o/ppl${SEQ}.log"
  ppl_rc=${PIPESTATUS[0]}
  note "--- ppl@${SEQ} rc=$ppl_rc"
  if [ "$ppl_rc" -ne 0 ]; then
    note "FATAL PPL@${SEQ} failed rc=$ppl_rc -- a table row missing a PPL cell is how the"
    note "      2048-vs-4096 mixup happened in the first place. Not continuing."
    exit "$ppl_rc"
  fi
done
note "=== STAGE 3 DONE ==="

# ------------------------------------------------------------------ STAGE 4: union-9 zero-shot
# Invocation is byte-identical to _sparseforge_same_harness_21.sh's run_one()
# except `pretrained`. `--batch_size auto` is deliberate and is what the existing
# arms used: their results_*.json record batch_size="auto" with resolved
# batch_sizes=[64]. Hard-coding `--batch_size 64` would be a DIFFERENT
# invocation string from the four arms already in the table.
note "=== STAGE 4: union-9 zero-shot (9 tasks, all 8 GPUs) ==="
o=$OUT/$VARIANT
mkdir -p "$o/lm_eval_out"
CUDA_VISIBLE_DEVICES="$GPUS" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
"$LM_EVAL" \
  --model hf \
  --model_args "pretrained=$EXPDIR/$VARIANT,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
  --tasks $TASKS \
  --batch_size auto \
  --num_fewshot 0 \
  --output_path "$o/lm_eval_out" \
  --seed 0 \
  --trust_remote_code \
  --log_samples 2>&1 | tee "$o/lm_eval.log"
eval_rc=${PIPESTATUS[0]}
note "=== STAGE 4 DONE rc=$eval_rc ==="
if [ "$eval_rc" -ne 0 ]; then
  note "FATAL lm_eval failed rc=$eval_rc -- not aggregating a partial run"
  exit "$eval_rc"
fi

# ------------------------------------------------------------------ STAGE 5: aggregate
# aggregate_zeroshot_union9.py hard-fails if ANY of the 9 is absent (its line
# 103), so a 7-task row can never be silently averaged into a union-9 cell.
note "=== STAGE 5: aggregate (asserts all 9 tasks present) ==="
"$PY" "$AGG9" \
    --lm-eval-out "$o/lm_eval_out" \
    --output "$o/zeroshot_union9.json" \
    --model "sparseforge_tokenmatched_${ARM}_iter${ITER}_${VARIANT}" 2>&1 | tee -a "$PROG"
agg_rc=${PIPESTATUS[0]}
if [ "$agg_rc" -ne 0 ]; then
  note "FATAL aggregation failed rc=$agg_rc -- row INVALID, not writing a summary"
  exit "$agg_rc"
fi
note "=== STAGE 5 DONE ==="

# ------------------------------------------------------------------ STAGE 6: 2:4 verify (POST)
note "=== STAGE 6: verify 2:4 $VARIANT (POST-inference) ==="
CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$VERIFY" \
    --model "$EXPDIR/$VARIANT" --sample-layers 12 --seed 0 2>&1 \
  | tee "$OUT/verify_2of4_${VARIANT}_post.log"
post_rc=${PIPESTATUS[0]}
note "=== STAGE 6 DONE rc=$post_rc ==="
if [ "$pre_rc" -ne "$post_rc" ]; then
  note "WARN 2:4 gate changed across inference (pre=$pre_rc post=$post_rc) -- investigate"
fi

# ------------------------------------------------------------------ SUMMARY
note "=== SUMMARY ==="
"$PY" - "$ARM" "$VARIANT" "$CK" "$ITER" "$OUT" "$ELIGIBLE_2OF4" "$pre_rc" "$post_rc" <<'PYEOF' 2>&1 | tee -a "$PROG"
import json, sys, pathlib
arm, variant, ck, it, out, elig, pre_rc, post_rc = sys.argv[1:9]
out = pathlib.Path(out)
S = {"arm": arm, "variant": variant, "source_ckpt": ck, "source_iter": it,
     "2of4_eligible": elig, "verify_pre_rc": int(pre_rc), "verify_post_rc": int(post_rc),
     "tasks": "boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa",
     "n_tasks": 9,
     "harness": "lm_eval 0.4.8, --model hf, dtype=bfloat16, parallelize=True, "
                "add_bos_token=False, --batch_size auto, --num_fewshot 0, --seed 0",
     "ppl": {}}

for seq in ("4096", "2048"):
    f = out / variant / f"ppl{seq}" / "ppl_metrics.json"
    if f.exists():
        d = json.load(open(f))
        assert str(d["seqlen"]) == seq, f"seqlen mislabel: file says {d['seqlen']} in ppl{seq}/"
        S["ppl"][seq] = {"wikitext2_ppl": d["wikitext2_ppl"],
                         "wikitext2_tokens": d["wikitext2_tokens"],
                         "linear_zero_ratio": d["linear_zero_ratio"],
                         "exact_2of4_tile_ratio": d["exact_2of4_tile_ratio"]}
        print(f"  PPL @{seq}: {d['wikitext2_ppl']:.6f}  tokens={d['wikitext2_tokens']} "
              f"zero_ratio={d['linear_zero_ratio']:.9f} exact_2of4={d['exact_2of4_tile_ratio']}")
    else:
        print(f"  PPL @{seq}: MISSING ({f})")

z = out / variant / "zeroshot_union9.json"
if z.exists():
    b = json.load(open(z))
    S["per_task"] = {t: {"acc": e["acc"], "acc_norm": e["acc_norm"],
                         "primary_metric": e["primary_metric"], "n_samples": e["n_samples"]}
                     for t, e in b["per_task"].items()}
    for k in ("union9", "cast7", "ast7"):
        s = b[k]
        S[k] = {"mean_primary": s["mean_primary"], "mean_plain_acc": s["mean_plain_acc"]}
        print(f"  {k:7}: plain_acc {s['mean_plain_acc']*100:.4f}  primary {s['mean_primary']*100:.4f}")
    r = b["per_task"]["rte"]
    k_int = r["acc"] * r["n_samples"]
    S["rte_integral"] = {"acc": r["acc"], "n": r["n_samples"], "k": k_int,
                         "k_rounded": round(k_int)}
    print(f"  RTE: acc={r['acc']:.10f} n={r['n_samples']} k={k_int:.4f} -> {round(k_int)}/{r['n_samples']}")
    assert abs(k_int - round(k_int)) < 0.01, f"RTE k={k_int} is not integral -- metric mismatch"
    assert r["n_samples"] == 277, f"RTE n={r['n_samples']} != 277 -- wrong split"

(out / "tokenmatched_union9_summary.json").write_text(json.dumps(S, indent=2) + "\n")
print(f"  wrote {out/'tokenmatched_union9_summary.json'}")
if elig != "true":
    print("  ** 2:4 COLUMN: BARRED. This arm's export is not exact 2:4 "
          "(SLoRB folded). Report it, but never in a 2:4 column. **")
PYEOF
note "=== ALL STAGES COMPLETE (ARM=$ARM variant=$VARIANT) ==="
