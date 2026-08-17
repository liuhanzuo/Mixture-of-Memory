#!/usr/bin/env bash
# Paper B depth-ladder step200000 eval driver — ONE arm per invocation.
#
# Purpose: when keep8 / keep10 / keep12 reach step200000, produce the missing
# 200k row of the depth ladder under EXACTLY the battery that keep14 train-all
# and freeze_front used at 200k:
#     (1) held-out NTP PPL   (2) core-6 downstream   (3) knowledge-5 (incl MMLU)
# Same three harnesses, same order, same flags as
#   scripts/_run_olmo2_eval_keep14_s200000_b200.sh   (keep14 @200k)
#   scripts/_run_olmo2_eval_freezefront_s200000.sh   (freeze_front @200k)
# with the shard-completeness discipline of
#   scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh   (the clean `_v2` batteries)
# added on top, plus hard preflight assertions (see PROTOCOL doc).
#
# Protocol reference (read this before changing anything here):
#   paperB/LADDER_200K_EVAL_PROTOCOL.md
#
# Parameterised: nothing about a single arm is hardcoded. Everything comes from
# env vars so the same file serves keep8 / keep10 / keep12 (and any later rung).
#
# --------------------------------------------------------------------------
# USAGE
# --------------------------------------------------------------------------
#   ARM=keep12 \
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   CKPT=outputs/olmo2_probe2_7B_keep12fresh2/step200000.pt \
#   KEEP_FRONT=12 \
#   setsid nohup bash scripts/eval_paperb_ladder_200k.sh \
#     > logs/ladder200k_eval_keep12.log 2>&1 &
#
#   Add DRY_RUN=1 to run preflight + assertions ONLY (no GPU, no model load).
#   Add SKIP_ARCH_GUARD=1 only if you have a written reason; see PROTOCOL §4.
#
# Env vars (defaults in parentheses):
#   ARM            arm label, one of keep8|keep10|keep12                (required)
#   PROJECT_ROOT   node's repo root -- MUST be the disk holding CKPT    (required)
#   PYTHON_BIN     (/opt/conda/envs/torch-base/bin/python)  torch 2.13.0 on all 5 nodes
#   CKPT           ckpt path, relative to PROJECT_ROOT or absolute      (derived from ARM)
#   KEEP_FRONT     8|10|12                                             (derived from ARM)
#   N_FRESH        (2)
#   EXPECT_STEP    (200000)
#   BASE_MODEL     (../models/OLMo-2-1124-7B)
#   VAL            (data/dolmino_now_val.npy)
#   BS_PPL         (4)      keep14/freezefront @200k used 4
#   BS_MC          (8)      keep14/freezefront @200k used 8; base row's BS=16 is
#                           the off-protocol defect from PAPERB_FLIP_BOUNDARY_RESOLVED.md
#   NUM_SHARDS     (8)
#   NAME           (7B_${ARM}_step${EXPECT_STEP})     output-name base
#   REQUIRE_SM     (9.0)    required compute capability; see PROTOCOL §4
#   DRY_RUN        (0)
#   SKIP_ARCH_GUARD (0)
#
# --------------------------------------------------------------------------
# HARD PROTOCOL FACTS (do not "simplify" these away)
# --------------------------------------------------------------------------
# * chat_template=False is passed STRUCTURALLY, not as a flag: neither
#   eval_olmo2_probe2_ppl.py nor eval_olmo2_probe2_downstream.py contains the
#   string "chat_template" at all -- they score raw (context, continuation)
#   log-likelihoods and never call apply_chat_template. There is no flag to
#   pass and no way to turn it on. This script asserts that invariant by
#   grepping the harnesses, so a future edit that introduces a chat path
#   cannot silently change the protocol. (CLAUDE.md: chat=False is mandatory;
#   memory paper-eval-chat-false-mandatory.)
# * add_bos=0 (harness default; OLMo-2 is a BASE LM and its tokenizer does not
#   auto-add BOS -- eval_olmo2_probe2_downstream.py:27-28). Asserted post-merge.
# * fp32 master weights + bf16-autocast forward, inherited verbatim from the
#   harnesses. Not configurable here on purpose.
# * Per-task n_scored == pinned expected count is asserted AFTER every merge.
#   The downstream harness's own merge() does NOT refuse a partial shard set.
set -u
set -o pipefail

# ---------------- config from env ----------------
ARM="${ARM:?ARM is required (keep8|keep10|keep12)}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required (must be the disk holding CKPT)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
N_FRESH="${N_FRESH:-2}"
EXPECT_STEP="${EXPECT_STEP:-200000}"
BASE_MODEL="${BASE_MODEL:-../models/OLMo-2-1124-7B}"
VAL="${VAL:-data/dolmino_now_val.npy}"
BS_PPL="${BS_PPL:-4}"
BS_MC="${BS_MC:-8}"
NUM_SHARDS="${NUM_SHARDS:-8}"
REQUIRE_SM="${REQUIRE_SM:-9.0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ARCH_GUARD="${SKIP_ARCH_GUARD:-0}"

case "$ARM" in
  keep8)  DEF_KEEP=8;  DEF_DIR=outputs/olmo2_probe2_7B_keep8fresh2  ;;
  keep10) DEF_KEEP=10; DEF_DIR=outputs/olmo2_probe2_7B_keep10fresh2 ;;
  keep12) DEF_KEEP=12; DEF_DIR=outputs/olmo2_probe2_7B_keep12fresh2 ;;
  *) echo "FATAL: unknown ARM=$ARM (expected keep8|keep10|keep12). To add a rung, extend this case and paperB/LADDER_200K_EVAL_PROTOCOL.md." >&2; exit 2 ;;
esac
KEEP_FRONT="${KEEP_FRONT:-$DEF_KEEP}"
CKPT="${CKPT:-$DEF_DIR/step${EXPECT_STEP}.pt}"
NAME="${NAME:-7B_${ARM}_step${EXPECT_STEP}}"
NAMEK="${NAME}_know"

CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW_TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

ts () { date '+%F %T'; }
log () { echo "[$(ts)] $*"; }
die () { echo "[$(ts)] FATAL: $*" >&2; exit 2; }

cd "$PROJECT_ROOT" || die "cannot cd to PROJECT_ROOT=$PROJECT_ROOT"

ASSERT="$PYTHON_BIN scripts/_ladder200k_assert.py"
EVID_DIR="paperB/evidence"
mkdir -p logs olmo2_ppl_results olmo2_downstream_results "$EVID_DIR"

log "======== Paper B ladder ${ARM} @ step${EXPECT_STEP} eval ========"
log "PROJECT_ROOT=$PROJECT_ROOT"
log "PYTHON_BIN=$PYTHON_BIN"
log "CKPT=$CKPT  KEEP_FRONT=$KEEP_FRONT  N_FRESH=$N_FRESH"
log "NAME=$NAME  NAMEK=$NAMEK  BS_PPL=$BS_PPL  BS_MC=$BS_MC  NUM_SHARDS=$NUM_SHARDS"
log "DRY_RUN=$DRY_RUN"

# ==========================================================================
# PREFLIGHT (every check is fatal; all run before any GPU work)
# ==========================================================================
log "---- PREFLIGHT ----"

# P0a. NODE-EXCLUSIVE LOCK. This driver shards one 7B model across ALL 8 GPUs
#      (CUDA_VISIBLE_DEVICES=$g, one rank per card), so two concurrent invocations put
#      two 7B models on every card. That is exactly the OOM that destroyed 4 of 5 rungs
#      on 2026-08-08.
#
#      This is not hypothetical. Measured 2026-08-17: TWO independent watchers are armed
#      to launch this driver on the SAME node (.73):
#        - chain_keep12_eval_200k.sh (PID 1243702, on .73) fires when keep12's
#          step200000.pt lands; polls every 300 s.
#        - chain_keep10_ship_and_eval_200k.sh (PID 655909, on LOCAL) fires when .73
#          reports 0 compute PIDs for 2 consecutive polls, also every 300 s.
#      Neither has a lock, and grep for flock/lockfile/pgrep in this driver returned
#      nothing. When keep12's training exits there is a window with 0 compute PIDs
#      BEFORE keep12's own eval has claimed a card -- and a model-load phase reads 0 PIDs
#      too. Whether they collide therefore depends on which watcher samples first, i.e.
#      on a coin flip rather than on a design.
#
#      Both chains pass through THIS script, so one lock here guards both without editing
#      either running watcher (a running shell re-reads its .sh; editing one is a bug).
#
#      Fails closed and LOUD: a second arrival exits non-zero instead of queueing,
#      because the watcher that lost keeps polling and will retry, whereas a silent queue
#      would hide the scheduling defect. The lock is an flock held on fd 9 for the life of
#      the process, so kill -9 on the HOLDER releases it -- verified, rc=0 on the next
#      attempt. (Caveat learned the hard way: killing a wrapper shell does NOT release it,
#      because the child that owns fd 9 survives. When diagnosing a stuck lock, find the
#      real owner via /proc/*/fd/9 -> the lock path, not via the launching pid.)
LOCK_DIR="${LOCK_DIR:-$PROJECT_ROOT/.locks}"
mkdir -p "$LOCK_DIR" 2>/dev/null || true
LOCK_FILE="$LOCK_DIR/ladder200k_eval_node.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$LOCK_FILE" || die "cannot open lock file $LOCK_FILE"
  if ! flock -n 9; then
    # tail -1, NOT head -1: the file is APPENDED to by every holder, so head names the
    # FIRST-EVER holder, which after the first run is a long-dead pid. Measured
    # 2026-08-17 in the lock's own controls -- the refusal message named pid 1030590
    # while the real holder was 1030610. Same defect class as reading the head of any
    # append-only record; the current truth is at the end.
    holder=$(tail -1 "$LOCK_FILE" 2>/dev/null)
    die "REFUSING: another ladder eval already holds this node's GPUs.
     lock=$LOCK_FILE holder=${holder:-unknown}
     This driver needs all 8 cards; two at once OOMs both (2026-08-08 cost 4/5 rungs).
     The losing watcher should keep polling and retry -- this is a REFUSAL, not a crash."
  fi
  printf 'ARM=%s pid=%s host=%s started=%s\n' "$ARM" "$$" "$(hostname)" "$(date -Is)" >&9
  log "P0a node lock acquired: $LOCK_FILE (ARM=$ARM pid=$$)"
else
  # Never silently proceed unguarded -- otherwise the log implies a protection that is
  # not present.
  log "P0a WARNING: flock(1) not found; running WITHOUT the node-exclusive lock. A second
     ladder eval on this node would OOM both. Check by hand: pgrep -af eval_paperb_ladder_200k"
fi

# P0. interpreter + torch version. torch 2.7.0 vs 2.13.0 moved ~20 items on
#     bit-identical weights (status/PAPERB_FLIP_BOUNDARY_RESOLVED.md). All the
#     clean `_v2` batteries are torch 2.13.0; the stale olmo2_venv is 2.7.0.
[ -x "$PYTHON_BIN" ] || die "PYTHON_BIN not executable: $PYTHON_BIN"
TORCH_VER=$(CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -c 'import torch;print(torch.__version__)' 2>/dev/null) \
  || die "cannot import torch with $PYTHON_BIN"
log "P0 torch=$TORCH_VER"
case "$TORCH_VER" in
  2.13.*) : ;;
  *) die "torch=$TORCH_VER but the clean single-protocol batteries are torch 2.13.x. A version change alone moves ~20 items (PAPERB_FLIP_BOUNDARY_RESOLVED.md). Use /opt/conda/envs/torch-base/bin/python." ;;
esac

# P1. harness scripts present.
for f in scripts/eval_olmo2_probe2_ppl.py scripts/eval_olmo2_probe2_downstream.py \
         scripts/_ladder200k_assert.py; do
  [ -f "$f" ] || die "missing harness/helper: $PROJECT_ROOT/$f (two-disk trap: scp -O it over)"
done
log "P1 harnesses present"

# P2. chat_template invariant. Neither harness may reference a chat template.
for f in scripts/eval_olmo2_probe2_ppl.py scripts/eval_olmo2_probe2_downstream.py; do
  if grep -q "apply_chat_template\|chat_template" "$f"; then
    die "$f now references a chat template. Paper B is a BASE-LM protocol (chat_template=False is mandatory, CLAUDE.md). Verify no chat path is reachable before re-enabling this eval."
  fi
done
log "P2 chat_template=False verified structurally (no chat-template code path in either harness)"

# P3. base model + validation array.
[ -d "$BASE_MODEL" ] || die "BASE_MODEL dir not found: $BASE_MODEL (relative to $PROJECT_ROOT)"
[ -f "$VAL" ] || die "VAL not found: $VAL"
VAL_MD5=$(md5sum "$VAL" | awk '{print $1}')
log "P3 base_model=$BASE_MODEL  val=$VAL md5=$VAL_MD5"
if [ "$VAL_MD5" != "f2ea48a2074a2f38fc3b4477fceecf11" ]; then
  die "held-out val array md5=$VAL_MD5 but every ladder PPL on record was measured on f2ea48a2074a2f38fc3b4477fceecf11 (verified identical on wzc1 and zwfy6, 2026-08-15). A different val array makes the PPL column non-comparable."
fi

# P4. GPU architecture guard. The clean single-protocol Table 4 is H20 cc9.0.
#     core6 has a 0.03-0.16 pp cross-architecture floor on bit-identical weights
#     (status/PAPERB_CORE6_CROSSARCH_FLOOR.md) -- small, but the ladder must not
#     mix. See paperB/LADDER_200K_EVAL_PROTOCOL.md §4 for the full argument.
CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',' )
log "P4 compute_cap(s) on this node: ${CAPS:-<none>}"
if [ "$SKIP_ARCH_GUARD" != "1" ]; then
  case "$CAPS" in
    "${REQUIRE_SM},") : ;;
    *) die "this node reports compute_cap=[${CAPS%,}] but the ladder protocol requires ${REQUIRE_SM} (H20). Table 4's clean single-protocol batteries are all H20 cc9.0; core6 has a 0.03-0.16pp cross-arch floor on bit-identical weights. Run on .73/.82/.104, or set SKIP_ARCH_GUARD=1 AND record the deviation in the protocol doc." ;;
  esac
else
  log "P4 WARNING: SKIP_ARCH_GUARD=1 -- this run will NOT be architecture-comparable to the existing ladder rows. Record why."
fi

# P5. ckpt exists, is at EXPECT_STEP, and its arch meta matches the arm.
CKMETA="$EVID_DIR/ladder200k_${ARM}_ckptmeta.json"
$ASSERT ckpt --path "$CKPT" \
  --expect-step "$EXPECT_STEP" --expect-keep "$KEEP_FRONT" --expect-fresh "$N_FRESH" \
  --out "$CKMETA" || die "ckpt preflight failed for $CKPT"
log "P5 ckpt assertions passed; meta -> $CKMETA"

# P6. ckpt md5 recorded for provenance. Full md5 of a 34-44 GiB file costs
#     ~1.5-3 min at the measured ~500 MB/s hashing rate; that is cheap relative
#     to the ~1 h eval, and it is the only way to prove later that the numbers
#     came from these bytes. Set CKPT_MD5_MODE=head to hash only the first 2 GiB.
CKPT_MD5_MODE="${CKPT_MD5_MODE:-full}"
case "$CKPT_MD5_MODE" in
  full) CKPT_MD5=$(md5sum "$CKPT" | awk '{print $1}'); CKPT_MD5_SCOPE=full ;;
  head) CKPT_MD5=$(dd if="$CKPT" bs=8M count=256 2>/dev/null | md5sum | awk '{print $1}'); CKPT_MD5_SCOPE=first2GiB ;;
  none) CKPT_MD5="skipped"; CKPT_MD5_SCOPE=none ;;
  *) die "CKPT_MD5_MODE must be full|head|none (got $CKPT_MD5_MODE)" ;;
esac
log "P6 ckpt md5($CKPT_MD5_SCOPE)=$CKPT_MD5"

# P7. output dirs must not already hold a merged battery -- never overwrite an
#     existing number. (The harness itself would happily clobber summary.json.)
for pair in "olmo2_ppl_results/$NAME" "olmo2_downstream_results/$NAME" "olmo2_downstream_results/$NAMEK"; do
  if [ -f "$pair/summary.json" ]; then
    die "$pair/summary.json already exists. Refusing to overwrite an existing measurement. Pick a different NAME (e.g. NAME=${NAME}_v2) or move the old dir aside."
  fi
  if [ -d "$pair" ] && [ -n "$(ls -A "$pair" 2>/dev/null)" ]; then
    log "P7 WARNING: $pair exists and is non-empty (no summary.json). Stale shard files from a crashed run will be re-used by --merge. Contents:"
    ls -la "$pair" | sed 's/^/    /'
    die "clean or rename $pair before running, so the merge cannot mix old and new shards."
  fi
done
log "P7 output dirs clean (no pre-existing summary.json)"

# P8. free disk. A battery writes only JSON/JSONL, but a full ceph is how the
#     keep14 run once died mid-save.
AVAIL_KB=$(df -k . | awk 'NR==2{print $4}')
log "P8 free space on $PROJECT_ROOT: $((AVAIL_KB/1024/1024)) GiB"
[ "$AVAIL_KB" -gt 5242880 ] || die "less than 5 GiB free on $PROJECT_ROOT"

log "---- PREFLIGHT PASSED (8/8) ----"

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN=1 -> stopping before any GPU work. No model was loaded, no GPU touched."
  cat > "$EVID_DIR/ladder200k_${ARM}_preflight.json" <<EOF
{
  "arm": "$ARM",
  "dry_run": true,
  "ts": "$(ts)",
  "project_root": "$PROJECT_ROOT",
  "python_bin": "$PYTHON_BIN",
  "torch": "$TORCH_VER",
  "compute_caps": "${CAPS%,}",
  "ckpt": "$CKPT",
  "ckpt_md5": "$CKPT_MD5",
  "ckpt_md5_scope": "$CKPT_MD5_SCOPE",
  "expect_step": $EXPECT_STEP,
  "keep_front_layers": $KEEP_FRONT,
  "n_fresh_layers": $N_FRESH,
  "val": "$VAL",
  "val_md5": "$VAL_MD5",
  "bs_ppl": $BS_PPL,
  "bs_mc": $BS_MC,
  "num_shards": $NUM_SHARDS,
  "names": {"ppl": "$NAME", "core6": "$NAME", "know5": "$NAMEK"},
  "chat_template": false,
  "add_bos": 0
}
EOF
  log "preflight record -> $EVID_DIR/ladder200k_${ARM}_preflight.json"
  exit 0
fi

# ==========================================================================
# GPU environment (matches the keep14/freezefront @200k drivers verbatim)
# ==========================================================================
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/hf_datasets_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_DATASETS_CACHE"

# assert_shards <results_root> <name> -- every shard{i}of{N}.json must exist
# BEFORE the merge. A partial merge is silent contamination: keep12's published
# arc_easy came from 6/8 shards and moved the cell +0.19 pp.
assert_shards () {
  local RROOT=$1 NM=$2 MISS=0 g
  local D="$RROOT/$NM"
  for ((g=0; g<NUM_SHARDS; g++)); do
    [ -f "$D/shard${g}of${NUM_SHARDS}.json" ] || { echo "[$(ts)]   SHARD MISSING: $D/shard${g}of${NUM_SHARDS}.json" >&2; MISS=$((MISS+1)); }
  done
  [ "$MISS" -eq 0 ] || die "$NM: $MISS/$NUM_SHARDS shard files missing. Refusing to merge (a partial merge changes the measurement basis). Inspect logs/*_${NM}_shard*.log, re-run the failed shards, then re-invoke."
}

# ---------- (1) held-out PPL ----------
log "---- (1) held-out PPL -> olmo2_ppl_results/$NAME ----"
for ((g=0; g<NUM_SHARDS; g++)); do
  CUDA_VISIBLE_DEVICES=$g "$PYTHON_BIN" scripts/eval_olmo2_probe2_ppl.py \
    --base_model "$BASE_MODEL" --ckpt "$CKPT" \
    --keep_front_layers "$KEEP_FRONT" --n_fresh_layers "$N_FRESH" \
    --val_path "$VAL" --num_shards "$NUM_SHARDS" --shard_index $g \
    --batch_size "$BS_PPL" --output_name "$NAME" \
    > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
done
wait
assert_shards olmo2_ppl_results "$NAME"
"$PYTHON_BIN" scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1
$ASSERT battery --results-root olmo2_ppl_results --name "$NAME" --kind ppl --num-shards "$NUM_SHARDS" \
  || die "PPL battery incomplete for $NAME"
log "(1) PPL summary:"; cat "olmo2_ppl_results/$NAME/summary.json"; echo

# ---------- (2) core-6 downstream ----------
log "---- (2) core-6 -> olmo2_downstream_results/$NAME ----"
"$PYTHON_BIN" scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS" \
  > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1 || true
for ((g=0; g<NUM_SHARDS; g++)); do
  CUDA_VISIBLE_DEVICES=$g "$PYTHON_BIN" scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE_MODEL" --ckpt "$CKPT" --tasks "$CORE_TASKS" \
    --keep_front_layers "$KEEP_FRONT" --n_fresh_layers "$N_FRESH" \
    --num_shards "$NUM_SHARDS" --shard_index $g --batch_size "$BS_MC" \
    --save_per_example --output_name "$NAME" \
    > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
done
wait
assert_shards olmo2_downstream_results "$NAME"
"$PYTHON_BIN" scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
$ASSERT battery --results-root olmo2_downstream_results --name "$NAME" --kind core6 --num-shards "$NUM_SHARDS" \
  || die "core-6 battery incomplete for $NAME"
log "(2) core-6 summary:"; head -c 1200 "olmo2_downstream_results/$NAME/summary.json"; echo

# ---------- (3) knowledge-5 ----------
log "---- (3) knowledge-5 -> olmo2_downstream_results/$NAMEK ----"
"$PYTHON_BIN" scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$KNOW_TASKS" \
  > "logs/olmo2_downstream_${NAMEK}_prepare.log" 2>&1 || true
for ((g=0; g<NUM_SHARDS; g++)); do
  CUDA_VISIBLE_DEVICES=$g "$PYTHON_BIN" scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE_MODEL" --ckpt "$CKPT" --tasks "$KNOW_TASKS" \
    --keep_front_layers "$KEEP_FRONT" --n_fresh_layers "$N_FRESH" \
    --num_shards "$NUM_SHARDS" --shard_index $g --batch_size "$BS_MC" \
    --save_per_example --output_name "$NAMEK" \
    > "logs/olmo2_downstream_${NAMEK}_shard${g}.log" 2>&1 &
done
wait
assert_shards olmo2_downstream_results "$NAMEK"
"$PYTHON_BIN" scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAMEK" 2>&1
$ASSERT battery --results-root olmo2_downstream_results --name "$NAMEK" --kind know5 --num-shards "$NUM_SHARDS" \
  || die "knowledge-5 battery incomplete for $NAMEK"
log "(3) knowledge-5 summary:"; head -c 1200 "olmo2_downstream_results/$NAMEK/summary.json"; echo

# ---------- provenance record ----------
REC="$EVID_DIR/ladder200k_${ARM}_run.json"
cat > "$REC" <<EOF
{
  "arm": "$ARM",
  "dry_run": false,
  "ts_done": "$(ts)",
  "node_hostname": "$(hostname)",
  "project_root": "$PROJECT_ROOT",
  "python_bin": "$PYTHON_BIN",
  "torch": "$TORCH_VER",
  "compute_caps": "${CAPS%,}",
  "ckpt": "$CKPT",
  "ckpt_md5": "$CKPT_MD5",
  "ckpt_md5_scope": "$CKPT_MD5_SCOPE",
  "step": $EXPECT_STEP,
  "keep_front_layers": $KEEP_FRONT,
  "n_fresh_layers": $N_FRESH,
  "val": "$VAL",
  "val_md5": "$VAL_MD5",
  "bs_ppl": $BS_PPL,
  "bs_mc": $BS_MC,
  "num_shards": $NUM_SHARDS,
  "chat_template": false,
  "add_bos": 0,
  "outputs": {
    "ppl": "olmo2_ppl_results/$NAME/summary.json",
    "core6": "olmo2_downstream_results/$NAME/summary.json",
    "know5": "olmo2_downstream_results/$NAMEK/summary.json"
  }
}
EOF
log "provenance -> $REC"
log "======== ${ARM} @ step${EXPECT_STEP} eval ALL DONE (3/3 batteries, all assertions passed) ========"
