#!/usr/bin/env bash
# ============================================================================
# B05 NATIVE-READOUT eval — the 16-cell grid of PHASE_SEPARATION_PREREG.md §1.
#
# FORK PROVENANCE (stated so a reviewer can diff it):
#   source : proposal/backlog/A02-comem-write-read-repair/code/run_a02_read_tax_eval.sh
#   sha256 : 00123b51c6aaf06713f1078f58a042679a30a51fbc80fe43eee722c3e50d5675
#            (the CURRENT wzc1 bytes, verified at fork time 2026-08-17)
#   ⚠️ B05 STATUS.json:112 records the fork source as sha256 295f6f56ac42... .
#   That is the PREVIOUS committed revision (git 135707b). Commit c272ee3 changed
#   3 lines, all `proposal/active/A02-...` -> `proposal/backlog/A02-...` after the
#   A02 archive move. The asset is FINE; the recorded hash is stale. Do NOT
#   "restore" the 295f6f56 bytes — that reintroduces paths that no longer exist.
#
# WHAT THIS RUNS (prereg §1) — single variable: READOUT CAPACITY
# --------------------------------------------------------------
#   arm   resume_j  adapter   output dir (pre-declared, b05_phase_assign.py:99)
#   N6    6         NONE      ruler_results/b05_native_ruler_N6_j6
#   N9    9         NONE      ruler_results/b05_native_ruler_N9_j9
#   N12   12        NONE      ruler_results/b05_native_ruler_N12_j12
#   N18   18        NONE      ruler_results/b05_native_ruler_N18_j18
#
# Each arm is paired against an A02 LoRA arm ALREADY ON DISK (never re-run here;
# re-running would overwrite the anchors the paired deltas are computed against):
#   N6 <- A2 a02_rtax_ruler_A2_j6      N12 <- A4 a02_ruler_c2_j12_readlora
#   N9 <- A3 a02_rtax_ruler_A3_j9      N18 <- A5 a02_rtax_ruler_A5_j18
#   ceiling anchor for both readouts: A0 a02_dvr_ruler_j0_top12 (j=0, no adapter)
#
#   4 arms x 4 cells = 16 cells. ZERO training steps: the native readout has no
#   parameters to fit. That is why B05's first GPU spend is ~3.3 GPU-h, not ~192.
#
# THE ONE DIFFERENCE FROM THE FORK SOURCE THAT MATTERS
# ----------------------------------------------------
# A02's GATE A/GATE B verified adapter sha / rank / layer-span. B05 has NO
# ADAPTERS, so those gates are meaningless here and are REPLACED, not dropped:
#   GATE A' — assert no --lora_adapter reaches any arm (the single variable).
#   GATE B' — prereg §5's verify-before-dispatch: all 5 comparator dirs must
#             carry 8/8 *.records.json BEFORE any GPU is touched. The prereg
#             could not check this from wzc1 (/apdcephfs_zwfy6 is not mounted
#             there); this script checks it on the run node, fail-closed.
#   GATE F  — prereg §4 item 5: the adjudicator's own --selftest must pass on
#             THIS node BEFORE the data exists. "No excuse for running it after
#             the fact." Fail-closed.
#
# PROTOCOL INVARIANTS (violating any voids the result, prereg §1)
#   * chat_template=False everywhere (base LM, no SFT/RL). Both eval scripts
#     default to False, so we pass nothing and the analyzer asserts it (GATE D).
#   * selector=iter_bm25, topk=12, iter_hop_topk=4, sink_tokens=bos,
#     chunk_size=512 — byte-identical to the A02 comparators, so pairing is by
#     construction rather than by post-hoc alignment.
#   * 8 shards, limit 100 — matches the comparators.
#   * BASE model string is the IDENTICAL string to the A02 anchors:
#     `../models/Qwen--Qwen3-8b`. Per CLAUDE.md that symlink resolves to
#     Qwen3-8B-*Instruct*, not -Base. That is DELIBERATE here and must not be
#     "fixed": B05's native column is paired item-by-item against A02 arms
#     measured with this exact string. Changing it breaks the pairing, which is
#     the whole design. Any base-vs-instruct question belongs to A02, not B05.
#   * shard completeness is asserted before a cell is accepted; a partial cell
#     ABORTS the arm rather than being merged (the line-174 gate below, which was
#     negative-tested in A02: deleting a shard produced G1_SHARD_INCOMPLETE 7/8).
#
# NODE: sm_90 H20 = .73 / .82 / .104, zwfy6 disk (prereg §5, three reasons: the
# comparator was measured on sm_90; the comparator raw cells live on zwfy6; the
# analyzer is sha-verified there). NOT B200 — a same-harness paired comparison
# must not straddle architectures.
#
# GPU POOL = all 8 by default. Override NGPU_POOL if another job holds cards;
# check `nvidia-smi` first.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU_POOL BASE_MODEL
# Usage (on .73 / .82 / .104):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/backlog/B05-semantic-handoff-phase-diagram/code/run_b05_native_eval.sh \
#     >logs/b05_native_eval.out 2>&1 &
#
# After ALL 16 cells are complete, the read-out is EXACTLY ONE invocation of
#   $PY proposal/backlog/B05-semantic-handoff-phase-diagram/code/b05_phase_assign.py \
#       --out proposal/backlog/B05-semantic-handoff-phase-diagram/evidence
# (prereg §4). This script prints that command at the end; it does NOT run it,
# because the read-out is a one-shot pre-registered commitment and must be a
# deliberate act, not a side effect of the driver finishing.
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd to $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NSHARD=8                          # must match the A02 comparators
POOL="${NGPU_POOL:-0 1 2 3 4 5 6 7}"
NPOOL=$(echo $POOL | wc -w)
BASE="${BASE_MODEL:-../models/Qwen--Qwen3-8b}"   # identical string to comparators
PROG=logs/b05_native_eval_progress.log
B05=proposal/backlog/B05-semantic-handoff-phase-diagram

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$W:$W/third_party/babilong-pkg:${PYTHONPATH:-}"
unset http_proxy https_proxy all_proxy
mkdir -p logs

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

note "==== B05 native-readout driver START on $(hostname) ===="
note "W=$W PY=$PY pool='$POOL' ($NPOOL gpus) nshard=$NSHARD base=$BASE"

# --- GATE A': the single variable. No adapter may reach any arm. -------------
# This is a source-level assertion on THIS FILE: the four dispatch lines below
# must contain no --lora_adapter. It is cheap and it is the invariant that
# defines B05 (native readout == the model's own layers, no learned repair).
SELF="$B05/code/run_b05_native_eval.sh"
# The pattern is written '--lora[_]adapter' ON PURPOSE. It matches the literal
# flag '--lora_adapter', but this line itself does NOT match it (after '--lora'
# comes '['), so the checker cannot indict itself. Writing the bare literal here
# made GATE A' fire on its own source: measured 4 self-matches before this fix.
LORA_RE='--lora[_]adapter'
if [ -f "$SELF" ]; then
  # Scope: the arm-dispatch lines only (a run_ruler_arm call, or its --resume_j
  # continuation). Those are the lines that can actually put a flag on a GPU.
  N_LORA_CODE=$(grep -v '^[[:space:]]*#' "$SELF" \
                | grep -E 'run_ruler_arm|--resume_j' \
                | grep -c -- "$LORA_RE" || true)
  if [ "$N_LORA_CODE" -ne 0 ]; then
    echo "FATAL GATE A': $N_LORA_CODE adapter flag(s) on an arm-dispatch line."
    echo "  B05's single variable is readout capacity; an adapter voids the arm."
    exit 6
  fi
  note "GATE A' PASS no adapter flag on any of the 4 arm-dispatch lines"
else
  note "GATE A' WARN cannot self-locate at $SELF; skipping source assertion"
fi

# --- GATE B': prereg §5 verify-before-dispatch, fail-closed -----------------
# All 5 comparator dirs must carry 8/8 *.records.json for the 4 primary cells.
# Without these, the pairing that every B05 number depends on is impossible, and
# the correct action is to DROP that rung and record the drop -- never to run the
# native arm anyway and pair it against nothing.
note "GATE B': verifying the 5 comparator dirs (prereg §5) ..."
$PY - "$W" "$NSHARD" <<'PYEOF' || { echo "FATAL GATE B': comparator evidence incomplete"; exit 8; }
import sys, glob, os
W, NSHARD = sys.argv[1], int(sys.argv[2])
# The 9-arm C2 pairing set of prereg §4 item 2 = 4 native + these 5.
COMPARATORS = {
    "A0": "a02_dvr_ruler_j0_top12",       # ceiling anchor, j=0 no adapter
    "A2": "a02_rtax_ruler_A2_j6",
    "A3": "a02_rtax_ruler_A3_j9",
    "A4": "a02_ruler_c2_j12_readlora",
    "A5": "a02_rtax_ruler_A5_j18",
}
CELLS = [("niah_multikey_1", "16k"), ("niah_multikey_1", "32k"),
         ("variable_tracking", "16k"), ("variable_tracking", "32k")]
bad = []
for arm, sub in COMPARATORS.items():
    d = os.path.join(W, "ruler_results", sub)
    if not os.path.isdir(d):
        bad.append(f"{arm}: DIR MISSING {d}")
        continue
    for task, length in CELLS:
        pat = os.path.join(d, f"{task}_{length}_shard*of{NSHARD}.records.json")
        n = len(glob.glob(pat))
        status = "OK" if n == NSHARD else "INCOMPLETE"
        print(f"  {arm:3s} {task:18s} {length:4s} {n}/{NSHARD} records  {status}")
        if n != NSHARD:
            bad.append(f"{arm}/{task}|{length}: {n}/{NSHARD} records.json")
if bad:
    print("GATE B' FAIL -- these rungs CANNOT be paired:")
    for b in bad:
        print("   " + b)
    print("Per prereg §5: drop the affected rung from the ladder AND record the")
    print("drop in PHASE_SEPARATION_PREREG.md / STATUS.json before running.")
    sys.exit(1)
print("GATE B' PASS: 5/5 comparator dirs x 4 primary cells at 8/8 records.")
PYEOF
note "GATE B' PASS comparator pairing evidence complete"

# --- GATE F: prereg §4 item 5 -- adjudicator selftest BEFORE the data exists -
note "GATE F: running the adjudicator selftest (0 GPU, no data needed) ..."
$PY "$B05/code/b05_phase_assign.py" --selftest || {
  echo "FATAL GATE F: adjudicator selftest FAILED on this node. The gate must be"
  echo "  demonstrably decidable BEFORE data exists (prereg §4 item 5)."
  exit 5
}
note "GATE F PASS adjudicator selftest green pre-data"

# --- pre-data assertion, recorded in the log for later audit ----------------
PREEXIST=$(ls -d ruler_results/b05_native_ruler_* 2>/dev/null | wc -l)
note "pre-data check: $PREEXIST existing b05_native_ruler_* dirs (prereg §4 expects 0 on a first run)"

# Retrieval flags: byte-identical to the A02 comparators on disk.
RETR="--selector iter_bm25 --topk 12 --iter_hop_topk 4 --sink_tokens bos"

# ---------------------------------------------------------------------------
# Identical in structure to the fork source's run_ruler_arm, INCLUDING the
# fail-closed abort (source line 174) which is the prereg §4 item 1 completeness
# gate.  It aborts the arm rather than merging a partial cell.
run_ruler_arm() {
  local NAME="$1" EXTRA="$2"; shift 2
  local TASKS="$1" LENS="$2"
  note "ruler $NAME START tasks='$TASKS' lens='$LENS'"
  local want=$(( $(echo $TASKS | wc -w) * $(echo $LENS | wc -w) * NSHARD ))
  local have; have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  if [ "$have" -eq "$want" ]; then note "  SKIP ruler $NAME ($have/$want records present)"; return 0; fi
  local slot=0
  for g in $POOL; do
    ( for s in $(seq 0 $((NSHARD-1))); do
        [ $((s % NPOOL)) -eq "$slot" ] || continue
        CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
          --model_path "$BASE" $EXTRA \
          --ruler_tasks $TASKS --lengths $LENS \
          --limit 100 --chunk_size 512 \
          --num_shards $NSHARD --shard_index "$s" \
          --output_name "$NAME" \
          > "logs/b05_native_ruler_${NAME}_shard${s}.log" 2>&1
      done ) &
    slot=$((slot+1))
  done
  wait
  have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  if [ "$have" -ne "$want" ]; then note "  ABORT ruler $NAME: only $have/$want records" >&2; return 9; fi
  note "ruler $NAME DONE ($have/$want records)"
}

RUL_TASKS="niah_multikey_1 variable_tracking"
RUL_LENS="16k 32k"

# ---- THE 16 CELLS: 4 native arms, no adapter, single variable = resume_j ----
# NOTE: no --lora_adapter on ANY line below.  GATE A' asserts this.
run_ruler_arm b05_native_ruler_N6_j6   "--resume_j 6  $RETR" "$RUL_TASKS" "$RUL_LENS" \
  || { note "FATAL: N6 aborted (incomplete shards); NOT continuing to a partial grid"; exit 9; }
run_ruler_arm b05_native_ruler_N9_j9   "--resume_j 9  $RETR" "$RUL_TASKS" "$RUL_LENS" \
  || { note "FATAL: N9 aborted (incomplete shards); NOT continuing to a partial grid"; exit 9; }
run_ruler_arm b05_native_ruler_N12_j12 "--resume_j 12 $RETR" "$RUL_TASKS" "$RUL_LENS" \
  || { note "FATAL: N12 aborted (incomplete shards); NOT continuing to a partial grid"; exit 9; }
run_ruler_arm b05_native_ruler_N18_j18 "--resume_j 18 $RETR" "$RUL_TASKS" "$RUL_LENS" \
  || { note "FATAL: N18 aborted (incomplete shards); NOT continuing to a partial grid"; exit 9; }

note "==== ALL 4 NATIVE ARMS COMPLETE (16 cells) ===="

# --- completeness restated, then STOP. The read-out is a deliberate act. -----
TOT=0
for d in b05_native_ruler_N6_j6 b05_native_ruler_N9_j9 \
         b05_native_ruler_N12_j12 b05_native_ruler_N18_j18; do
  n=$(ls ruler_results/"$d"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  note "  $d: $n/32 records.json"
  TOT=$((TOT + n))
done
note "total native records.json = $TOT / 128 (4 arms x 4 cells x 8 shards)"
if [ "$TOT" -ne 128 ]; then
  note "FATAL: grid incomplete ($TOT/128). Read-out MUST NOT be attempted."
  exit 9
fi

note "NEXT — the ONE-SHOT pre-registered read-out (prereg §4). Run it once:"
note "  $PY $B05/code/b05_phase_assign.py --out $B05/evidence"
note "It re-checks GATE C (8/8 + n=100 + no dup ids), GATE C2 (input_ids_sha256"
note "pairing across all 9 arms), GATE D (config identity, lora_adapter is None"
note "for all 4 native arms) and GATE E (canonical A02 loaders imported), then"
note "calls adjudicate() exactly once and writes the evidence JSON + verdict."
note "==== B05 native-readout driver DONE ===="
