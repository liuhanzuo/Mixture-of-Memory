#!/usr/bin/env bash
# ============================================================================
# RE-ARM the two SparseForge token-matched union-9 watchers after the
# 2026-08-13 22:2x node restart.
#
#   bash scripts/_rearm_sparseforge_union9_watchers.sh          # arm both
#   ARMS=noslorb bash scripts/_rearm_sparseforge_union9_watchers.sh
#
# The restart killed BOTH training arms and BOTH watchers, and destroyed the
# pinned harness stack. Training was resumed faithfully (commit 3db8c93,
# noslorb from iter_num=6700, slorb from 6500, both with optimizer state), so
# only the scoring half needs rebuilding. This wrapper re-arms
# _run_sparseforge_tokenmatched_union9_watcher.sh with THREE corrections that the
# original invocation can no longer satisfy.
#
# ---------------------------------------------------------------------------
# CORRECTION 1 — the harness lives in a venv now, not in conda
# ---------------------------------------------------------------------------
# The watcher defaults to /opt/conda/envs/torch-base/{python,lm_eval} and
# hard-asserts lm_eval 0.4.8 + transformers 4.57.6 (its lines 245-265, exit 21).
# That conda env now has transformers 5.15.0 and NO lm_eval, and it is the env the
# two live training arms are running in — so it must NOT be "fixed" to satisfy the
# assertion. Instead the pinned stack was rebuilt in an isolated venv,
# $ROOT/venv_union9, and injected via the watcher's own PYTHON_BIN / LM_EVAL_BIN
# overrides (the escape hatch its line 238 advertises). The assertion is left
# untouched and still has to pass on its own terms.
#
# ---------------------------------------------------------------------------
# CORRECTION 2 — the TOPOLOGY changed; the old TRAIN_NODE mapping is now BACKWARDS
# ---------------------------------------------------------------------------
# The watcher was written when slorb trained on the same box that scored, and
# noslorb trained elsewhere:
#     slorb   -> TRAIN_NODE=local  (pgrep is authoritative)
#     noslorb -> TRAIN_NODE=remote (use shared-log mtime)
# After the restart that is INVERTED. Verified from the live resume processes:
#     noslorb  trains HERE   (LOCAL, pid 40343 torchrun + 8 ranks,
#                             --out_dir out_llama_tokenmatched_noslorb)
#     slorb    trains on .212 (28.89.18.212), a DIFFERENT box on the SAME wzc1 disk
# Note LOCAL *is* 28.89.19.21 (its own `hostname -I` lists 28.89.19.21), so the
# watcher header's "both scored on .21" is still satisfied by running here — but
# leaving ARM=slorb on TRAIN_NODE=local would make it pgrep for a trainer that
# lives on another host, find nothing, and declare training finished ~7 h early.
# So TRAIN_NODE and TRAIN_LOG are passed explicitly per arm.
#
# Liveness stays exactly as the watcher designed it and is NEVER instantaneous
# nvidia-smi utilisation (memory [[one-sample-is-not-a-trend-or-state]]):
#   noslorb  process existence via pgrep on this node (authoritative)
#   slorb    shared-log mtime younger than LOG_STALE_S=1800 (a 30-min duration
#            test on a log that advances ~1 line/45-55 s)
#
# ---------------------------------------------------------------------------
# CORRECTION 3 — GPU split follows who is actually free
# ---------------------------------------------------------------------------
# Unchanged from the watcher's design (0-3 / 4-7, matching every archived arm's
# 4-GPU parallelize=True topology). Both watchers score on LOCAL. They will not
# start until their own 4 cards are free: the watcher's complete() refuses while
# >GPU_FREE_MIB is held (its line 415), so neither can preempt a trainer.
#
# noslorb's trainer is ON LOCAL, so its cards free themselves when it exits.
# slorb's trainer is on .212, so LOCAL's 4-7 free up when noslorb's trainer exits
# too — meaning slorb's watcher may become GPU-ready before its ckpt is ready. That
# is fine and is the correct ordering: the ckpt guard (trainer_alive -> mtime ->
# size settle -> torch.load probe -> arm-conditional SLoRB tensor assertion) is
# what gates scoring, not card availability.
#
# ---------------------------------------------------------------------------
# ADMISSION GATE — will not arm on an unvalidated harness
# ---------------------------------------------------------------------------
# Refuses to launch unless the rebuilt stack has PASSED the same-arm reproduction
# control (outputs/union9_harness_rebuild_control/dense_ref/harness_rebuild_control.json
# with verdict PASS). Matching version strings are not evidence that the rebuild
# reproduces the archive; the -0.346 pp AST-7 retraction is what happens when
# that assumption is skipped. A stack that cannot reproduce dense_ref must not be
# used to add a row, so this wrapper hard-exits instead.
# ============================================================================
set -u

ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
MOM=$ROOT/Mixture-of-Memory
WATCHER=$MOM/scripts/_run_sparseforge_tokenmatched_union9_watcher.sh
CONTROL_JSON=$ROOT/outputs/union9_harness_rebuild_control/dense_ref/harness_rebuild_control.json

VENV=$ROOT/venv_union9
PY=$VENV/bin/python
LM_EVAL=$VENV/bin/lm_eval

ARMS="${ARMS:-noslorb slorb}"

# Live training logs from the 2026-08-14 faithful resume (commit 3db8c93).
LOG_NOSLORB=logs/sparseforge_tm_noslorb_RESUME_0814_123843.log
LOG_SLORB=logs/sparseforge_tm_slorb_RESUME_0814_124313.log

say() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }

[ -f "$WATCHER" ] || { say "FATAL watcher missing: $WATCHER"; exit 3; }
[ -x "$PY" ]      || { say "FATAL python missing: $PY"; exit 3; }
[ -x "$LM_EVAL" ] || { say "FATAL lm_eval missing: $LM_EVAL"; exit 3; }

# ---- ADMISSION GATE ----
if [ ! -f "$CONTROL_JSON" ]; then
  say "REFUSING TO ARM: no harness-rebuild control at"
  say "  $CONTROL_JSON"
  say "Run scripts/_union9_harness_rebuild_control.sh first. The rebuilt stack has"
  say "not been shown to reproduce an archived arm, so any row it produces would be"
  say "a cross-harness number of unknown offset."
  exit 20
fi
verdict=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('verdict','MISSING'))" \
          "$CONTROL_JSON" 2>/dev/null || echo UNREADABLE)
if [ "$verdict" != "PASS" ]; then
  say "REFUSING TO ARM: harness-rebuild control verdict = $verdict (need PASS)"
  "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  flips=%s doc_hash_problems=%s task_hash_mismatches=%s' % (
    d.get('flips'), d.get('total_doc_hash_problems'), d.get('task_hash_mismatches')))
for s in d.get('structural_errors', [])[:8]:
    print('  ! %s' % s)
" "$CONTROL_JSON" 2>/dev/null || true
  say "A rebuilt stack that cannot reproduce dense_ref must not add a union-9 row."
  exit 20
fi
say "admission gate OK: harness-rebuild control verdict=PASS"

for ARM in $ARMS; do
  case "$ARM" in
    noslorb) GPUS=0,1,2,3; TRAIN_NODE=local;  TRAIN_LOG=$LOG_NOSLORB ;;
    slorb)   GPUS=4,5,6,7; TRAIN_NODE=remote; TRAIN_LOG=$LOG_SLORB ;;
    *) say "FATAL unknown arm: $ARM"; exit 2 ;;
  esac
  OUT=$MOM/logs/sparseforge_tm_union9_${ARM}_watch_REARM.out
  say "arming ARM=$ARM gpus=$GPUS train_node=$TRAIN_NODE log=$TRAIN_LOG"
  ARM="$ARM" GPUS="$GPUS" TRAIN_NODE="$TRAIN_NODE" TRAIN_LOG="$TRAIN_LOG" \
  PYTHON_BIN="$PY" LM_EVAL_BIN="$LM_EVAL" \
    setsid nohup bash "$WATCHER" > "$OUT" 2>&1 &
  pid=$!
  say "  ARM=$ARM watcher pid=$pid log=$OUT"
  sleep 3
done

say "done. progress logs: $MOM/logs/sparseforge_tm_union9_{noslorb,slorb}_progress.log"
