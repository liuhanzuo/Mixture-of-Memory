#!/usr/bin/env bash
# ============================================================================
# Paper C P-C1 follow-up (task #133 + #165): DEPTH-SWEEP train+eval launcher.
#
# WHY: P-C1 established freeze-graft > from-scratch at ONE depth (keep14+fresh2 =
# 16L). This launcher extends the single point into a DEPTH CURVE to test whether
# the freeze-graft advantage is robust across depth. For each keep_front in
# {20,24,28} (n_fresh=2 -> 22/26/30L) it trains the selected arms, then scores
# SQuAD EM/F1 on each. Recipe copied VERBATIM from scripts/run_paperC_pc1.sh:
#   seq_len=2048, max_steps=1000, warmup_steps=150, eff_bs=128 (BS*GA*nGPU),
#   gradient_checkpointing=1, optimizer=adamw (fp32-master), seed=42.
#
# THREE ARMS (ARMS= selects; default "graft scratch" == the #133 pair):
#   graft   : freeze-graft (== P-C1 A4)  --freeze_front   lr=1e-4  lr_inh=2e-5
#   scratch : from-scratch (== P-C1 A3)  --from_scratch   lr=3e-4  lr_inh=3e-4
#   rtrunk  : random-trunk   (#165, NEW) --random_trunk   lr=3e-4  lr_inh=3e-4
#                                                         + --min_lr_inherited 1e-5
#
# ---- WHY `rtrunk` EXISTS (task #165) --------------------------------------
# A3 (--from_scratch) random-inits the trunk AND embed_tokens AND model.norm AND
# lm_head. A ~2M-token SQuAD SFT cannot learn a 100352-row vocab embedding plus
# an output map, so "A4 (inherits front-j) > A3" is CONFOUNDED: it may be entirely
# a readout-interface effect and say nothing about inheriting trunk weights.
# `rtrunk` (--random_trunk, added by task P0-5, 8/8 self-test in
# scripts/selftest_random_trunk.py) is the matched control: SAME depth/shape as
# graft, trunk randomly initialised, but embed_tokens / model.norm / lm_head are
# transplanted bit-identically from the base. So:
#     graft  vs rtrunk  = SINGLE VARIABLE: where do the trunk weights come from?
#     rtrunk vs scratch = SINGLE VARIABLE: is the readout interface inherited?
# and the A4-vs-A3 gap decomposes into those two legs.
#
# ---- rtrunk LR CHOICE: A3's 3e-4/3e-4, NOT A4's 1e-4/2e-5 ------------------
# rtrunk's trunk is randomly initialised, i.e. exactly A3's predicament (a random
# trunk needs a from-scratch LR; A4's 1e-4/2e-5 is tuned for *inherited* weights
# that must not be blown away). Pinning rtrunk to A3's LR makes `rtrunk vs
# scratch` a clean single-variable contrast (only the init of the 3 non-layer
# tensors differs, LR schedule byte-identical), and keeps the LR advantage from
# leaking into the graft-vs-rtrunk leg in the *conservative* direction.
#
# `_classify_param(random_trunk=True)` buckets model.layers.* -> 'fresh' (gets
# --lr) and embed_tokens / model.norm / lm_head -> 'inherited' (gets
# --lr_inherited). With lr == lr_inherited == 3e-4 every group sits on the same
# base LR, reproducing A3's single-LR schedule. ONE extra flag is needed for exact
# A3 parity: the cosine FLOORS differ by default (--min_lr 1e-5 vs
# --min_lr_inherited 2e-6), and A3/from_scratch puts everything in the 'fresh'
# bucket so it floors at 1e-5 everywhere -> rtrunk therefore also passes
# `--min_lr_inherited 1e-5`. graft/scratch do NOT get that flag (graft genuinely
# wants the 2e-6 floor on its inherited embed/norm; for scratch it is a no-op),
# so their command lines stay byte-identical to #133.
# NOTE for write-up: rtrunk DOES log two optim groups (fresh=trunk,
# inherited=readout) but they share base_lr AND min_lr -> it is a UNIFORM 3e-4
# cosine, do NOT describe it as differential LR.
#
# ---- DATA / VAL: the legacy SQuAD split is VOID ----------------------------
# `data/squad_val.jsonl` (old VAL_PATH default) and
# `data/squad_sft_olmo2_2048_train.npy` (old DATA_PATH default) are BOTH VOID as
# Historical capability experiment for the abandoned Paper C v1 proposal.
# See proposal/archive/paperC-v1-frozen-cap/scoping/SCOPING_AND_POSTMORTEM.md.
# forensics docstring of scripts/build_paperC_squad_eval.py:
#   * 997/2000 = 49.85% of legacy val target_text is one Chinese refusal string,
#     vs 17.56% in legacy train (32.29pp skew) -> an INPUT-BLIND CONSTANT scores
#     EM 49.85, above A4 (.2930) / A3 (.2605) / BASE (.3385).
#   * legacy `relevant_indices` is statistically indistinguishable from uniform
#     (z=+0.82) and refusal rows never had evidence removed -> "unanswerable" was
#     an inherited v2.0 label, not a design.
#   * VERIFIED 2026-08-06: `data/squad_sft_olmo2_2048_train.npy` is byte-identical
#     (md5 44f46c1d595e8f20d54ed9dcb6a9e34c) to re-tokenising the CONTAMINATED
#     legacy `data/squad_train.jsonl` -> the train side is contaminated too.
# Defaults therefore now point at the P0-4 rebuild (train/val SAME refusal rate,
# refusals made truly unanswerable by deleting gold-bearing chunks):
#     DATA_PATH=data/paperC_squad_v2/train_refusal25_olmo2_2048.npy   (1064 chunks)
#     VAL_PATH =data/paperC_squad_v2/val_refusal25.jsonl              (n=2000)
# The new .npy was produced (2026-08-06) by the UNMODIFIED tokeniser; regenerate
# it anywhere it is missing with
#     bash scripts/make_paperC_squad_v2_npy.sh
# (data/ is gitignored, so the .npy does NOT travel with the repo -- the zwfy6
#  nodes .73/.82 must run that script, or receive the file via `scp -O` + md5.
#  md5 of the refusal25 shard on wzc1 = 4be5e4e534213193fc5799cbc6c0b058.)
# (tiers 00/50 exist alongside it; the legacy .npy was NOT overwritten.)
#
# ****  CROSS-DATA COMPARISONS ARE INVALID  ****  #133's graft/scratch numbers
# were trained+scored on the VOID legacy split. The graft-vs-rtrunk contrast is
# only meaningful if BOTH arms use the SAME data, so a rtrunk run must be paired
# with graft (and scratch) RE-RUN on the clean split. To make that impossible to
# get wrong silently, OUT_DIR and the eval output_name now carry a DATA_TAG
# derived from DATA_PATH (and a val tag when the val tier differs).
# To reproduce #133 verbatim, pass the legacy paths explicitly:
#   DATA_PATH=$PWD/data/squad_sft_olmo2_2048_train.npy \
#   VAL_PATH=$PWD/data/squad_val.jsonl  ARMS="graft scratch" ...
#
# TARGET NODE: B200/L20A (183GB) or H20 (97.8GB, may need the BS ladder). scratch
# and rtrunk are full-param trains of a (keep+fresh)-layer model; 183GB fits
# fp32-AdamW, so OPT=adamw stays default for comparability with A3. If an arm
# OOMs, set OPT=bnb8bit (bitsandbytes 8-bit AdamW) and NOTE the optimizer
# difference in the report -- bnb is NOT installed on LOCAL/.252/.82.
#
# EVAL LOAD: each ckpt is a raw state_dict .pt from train_olmo2_arch_probe2 that
# STORES keep_front/n_fresh in its meta. eval_paperC_squad_emf1.py --ckpt rebuilds
# the exact (keep+fresh)-layer shell from that meta and strict-loads -> zero drift.
# We do NOT pass --keep_front_layers/--n_fresh_layers to the eval (meta drives it).
#
# FAULT-TOLERANT: each run is launched FOREGROUND and chained; a failed/OOM run is
# logged and the chain continues to the next. Eval only runs if final.pt exists.
#
# ****  DO NOT AUTO-RUN. This is only a launcher.  ****
#
# USAGE (task #165, the new arm only):
#   cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
#   ARMS=rtrunk KEEPS="20 24 28" PYBIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash scripts/run_paperC_depthsweep.sh \
#     > logs/paperC_depthsweep_rtrunk_sched.out 2>&1 &
#   # optional overrides:
#   #   KEEPS="20 24 28"  ARMS="graft scratch rtrunk"  GPUS=0,1,2,3,4,5,6,7
#   #   PORT=29560  MAX_STEPS=1000  EFF_BS=128  BS=4  OPT=adamw  EVAL_ONLY=0
#   #   DATA_PATH=...  VAL_PATH=...  DATA_TAG=...  OUT_SUFFIX=_dryrun
#   #   SAVE_EVERY=500  EVAL_LIMIT=0 (>0 -> --limit, smoke only)
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYBIN:-${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}}"
BASE="${BASE:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"

DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/paperC_squad_v2/train_refusal25_olmo2_2048.npy}"
VAL_PATH="${VAL_PATH:-$PROJECT_ROOT/data/paperC_squad_v2/val_refusal25.jsonl}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT:-29560}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-2048}"
WARMUP="${WARMUP:-150}"
SEED="${SEED:-42}"
EFF_BS="${EFF_BS:-128}"
BS="${BS:-4}"
FRESH="${FRESH:-2}"
SAVE_EVERY="${SAVE_EVERY:-500}"
OPT="${OPT:-adamw}"                         # adamw (== A3, default) | bnb8bit (OOM fallback)
read -r -a KEEPS <<< "${KEEPS:-20 24 28}"
read -r -a ARMS  <<< "${ARMS:-graft scratch}"   # graft | scratch | rtrunk
EVAL_ONLY="${EVAL_ONLY:-0}"                 # 1 -> skip training, just re-eval existing final.pt
EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"               # >0 -> eval only the first N val rows (SMOKE ONLY)
OUT_SUFFIX="${OUT_SUFFIX:-}"                # appended to OUT_DIR + output_name (e.g. _dryrun)

# ---- data tag: makes cross-data runs impossible to mix up silently ----------
# The legacy (VOID) split gets the historical bare names so #133 stays byte-for-
# byte reproducible in place; anything else gets a tag from the file stem.
_data_base="$(basename "$DATA_PATH")"
_val_base="$(basename "$VAL_PATH")"
if [ -z "${DATA_TAG+x}" ]; then
  case "$_data_base" in
    squad_sft_olmo2_2048_train.npy) DATA_TAG="" ;;                       # legacy == #133
    train_refusal*_olmo2_2048.npy)
      DATA_TAG="_$(printf '%s' "$_data_base" | sed 's/^train_\(refusal[0-9]*\)_olmo2_2048\.npy$/\1/')" ;;
    *) DATA_TAG="_$(printf '%s' "$_data_base" | sed 's/\.npy$//; s/[^A-Za-z0-9]\+/-/g')" ;;
  esac
  # if the val tier disagrees with the train tier, say so in the tag
  case "$_val_base" in
    val_refusal*.jsonl)
      _vt="$(printf '%s' "$_val_base" | sed 's/^val_\(refusal[0-9]*\)\.jsonl$/\1/')"
      [ "_$_vt" != "$DATA_TAG" ] && DATA_TAG="${DATA_TAG}_val${_vt}" ;;
    squad_val.jsonl) [ -n "$DATA_TAG" ] && DATA_TAG="${DATA_TAG}_vallegacy" ;;
  esac
fi
TAG="${DATA_TAG}${OUT_SUFFIX}"

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nGPU=$(awk -F, '{print NF}' <<< "$GPUS")
GA=$(( EFF_BS / (BS * nGPU) )); [ "$GA" -lt 1 ] && GA=1
REAL_EFF=$(( BS * GA * nGPU ))

LOGDIR="$PROJECT_ROOT/logs"; mkdir -p "$LOGDIR"
# keyed by TAG so a clean-split run can never clobber the #133 (legacy, TAG="")
# summary file. The table itself still globs ALL depthsweep_* results and prints
# each row's val set, so mixed-data rows are visible rather than silently merged.
SUMMARY_LOG="$LOGDIR/paperC_depthsweep_summary${TAG}.log"

log(){ echo "[depthsweep $(date '+%F %T')] $*"; }

log "PYBIN=$PYBIN nGPU=$nGPU BS=$BS GA=$GA eff_bs=$REAL_EFF (target $EFF_BS) OPT=$OPT max_steps=$MAX_STEPS"
[ "$REAL_EFF" -ne "$EFF_BS" ] && log "WARNING eff_bs=$REAL_EFF != $EFF_BS (adjust BS/EFF_BS)"
log "keeps=${KEEPS[*]} arms=${ARMS[*]} fresh=$FRESH  (DO-NOT-AUTORUN launcher; MAIN starts by hand)"
log "DATA_PATH=$DATA_PATH"
log "VAL_PATH=$VAL_PATH"
log "TAG='${TAG}'  (OUT_DIR/output_name carry it so clean-split runs never mix with the VOID legacy #133 runs)"

# ---- preflight: fail loudly rather than train on a missing/void file ---------
pf=0
for f in "$DATA_PATH" "$VAL_PATH" "$BASE/config.json" \
         "$PROJECT_ROOT/scripts/train_olmo2_arch_probe2.py" \
         "$PROJECT_ROOT/scripts/eval_paperC_squad_emf1.py"; do
  [ -e "$f" ] || { log "PREFLIGHT MISSING: $f"; pf=1; }
done
[ -x "$PYBIN" ] || { log "PREFLIGHT: PYBIN not executable: $PYBIN"; pf=1; }
for arm in "${ARMS[@]}"; do
  case "$arm" in graft|scratch|rtrunk) ;; *) log "PREFLIGHT: unknown arm '$arm' (want graft|scratch|rtrunk)"; pf=1 ;; esac
done
[ "$pf" = 0 ] || { log "FATAL preflight failed"; exit 1; }
case "$_data_base" in
  squad_sft_olmo2_2048_train.npy)
    log "WARNING: DATA_PATH is the VOID legacy npy (md5 44f46c1d...; == contaminated data/squad_train.jsonl). Only valid for reproducing #133." ;;
esac
case "$_val_base" in
  squad_val.jsonl)
    log "WARNING: VAL_PATH is the VOID legacy val (49.85% constant-refusal floor > every arm). Only valid for reproducing #133." ;;
esac
log "preflight OK"

OPT_FLAG=""
[ "$OPT" = "bnb8bit" ] && OPT_FLAG="--optimizer bnb_adamw8bit"
LIMIT_FLAG=""
[ "$EVAL_LIMIT" -gt 0 ] 2>/dev/null && LIMIT_FLAG="--limit $EVAL_LIMIT"

# per-arm recipe -----------------------------------------------------------------
#  graft / scratch : VERBATIM from run_paperC_pc1.sh A4/A3 -- byte-identical command
#                    lines to #133 (do not touch; #133 must stay reproducible).
#  rtrunk (#165)   : --random_trunk with A3's LR (random trunk == A3's predicament),
#                    plus --min_lr_inherited 1e-5 so the readout group's cosine
#                    FLOOR also matches A3's single 'fresh' bucket (default would
#                    be 2e-6). Net effect = uniform 3e-4 -> 1e-5 cosine everywhere.
#                    See the header for the full LR rationale.
arm_extra(){ case "$1" in
               graft)   echo "--freeze_front" ;;
               scratch) echo "--from_scratch" ;;
               rtrunk)  echo "--random_trunk --min_lr_inherited 1e-5" ;;
             esac; }
arm_lr(){    case "$1" in graft) echo "1e-4" ;; scratch) echo "3e-4" ;; rtrunk) echo "3e-4" ;; esac; }
arm_lrinh(){ case "$1" in graft) echo "2e-5" ;; scratch) echo "3e-4" ;; rtrunk) echo "3e-4" ;; esac; }

port_off=0
for K in "${KEEPS[@]}"; do
  for arm in "${ARMS[@]}"; do
    EXTRA="$(arm_extra "$arm")"; LR="$(arm_lr "$arm")"; LR_INH="$(arm_lrinh "$arm")"
    OUT_DIR="$PROJECT_ROOT/outputs/paperC_depthsweep_keep${K}_${arm}${TAG}"
    LOG_FILE="$LOGDIR/paperC_depthsweep_keep${K}_${arm}${TAG}.log"
    PORT=$(( PORT_BASE + port_off )); port_off=$(( port_off + 1 ))
    mkdir -p "$OUT_DIR"

    # ---- train (foreground, chained) ----
    if [ "$EVAL_ONLY" != "1" ] && [ ! -f "$OUT_DIR/final.pt" ]; then
      log "=== TRAIN keep$K/$arm KEEP=$K FRESH=$FRESH EXTRA='$EXTRA' LR=$LR LR_INH=$LR_INH PORT=$PORT -> $OUT_DIR ==="
      : > "$LOG_FILE"
      CUDA_VISIBLE_DEVICES="$GPUS" "$PYBIN" -m torch.distributed.run \
        --nnodes 1 --nproc_per_node "$nGPU" --rdzv_backend c10d --rdzv_endpoint "localhost:$PORT" \
        scripts/train_olmo2_arch_probe2.py \
          --data_path "$DATA_PATH" --output_dir "$OUT_DIR" --model_path "$BASE" \
          --keep_front_layers "$K" --n_fresh_layers "$FRESH" \
          --batch_size "$BS" --grad_accumulation_steps "$GA" --seq_len "$SEQ_LEN" \
          --lr "$LR" --lr_inherited "$LR_INH" --max_steps "$MAX_STEPS" \
          --warmup_steps "$WARMUP" --save_every "$SAVE_EVERY" --log_every 10 --seed "$SEED" \
          --gradient_checkpointing 1 $EXTRA $OPT_FLAG \
        >>"$LOG_FILE" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then log "FAIL train keep$K/$arm (exit $rc) -> continue"; fi
    else
      log "SKIP train keep$K/$arm (EVAL_ONLY=$EVAL_ONLY, final.pt exists=$( [ -f "$OUT_DIR/final.pt" ] && echo yes || echo no ))"
    fi

    # ---- eval SQuAD EM/F1 (same script/口径 as P-C1, val set per VAL_PATH) ----
    if [ -f "$OUT_DIR/final.pt" ]; then
      ONAME="depthsweep_keep${K}_${arm}${TAG}"
      log "=== EVAL keep$K/$arm SQuAD EM/F1 -> $ONAME (val=$_val_base) ==="
      # meta drives keep_front/n_fresh; we pass --base_model $BASE (cfg+tokenizer).
      CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | cut -d, -f1)" "$PYBIN" scripts/eval_paperC_squad_emf1.py \
        --ckpt "$OUT_DIR/final.pt" --base_model "$BASE" --tokenizer "$BASE" \
        --val_path "$VAL_PATH" --output_name "$ONAME" --batch_size "$EVAL_BATCH" \
        $LIMIT_FLAG \
        >>"$LOG_FILE" 2>&1 || log "FAIL eval keep$K/$arm -> continue"
      "$PYBIN" scripts/eval_paperC_squad_emf1.py --merge --output_name "$ONAME" \
        >>"$LOG_FILE" 2>&1 || log "FAIL merge keep$K/$arm -> continue"
    else
      log "no final.pt for keep$K/$arm -> skip eval"
    fi
  done
done

# ---- summary table (for MAIN to backfill) -----------------------------------
# ALSO prints the mandatory input-blind constant floor for VAL_PATH: any arm below
# it is not showing capability (that is exactly how the legacy split died).
log "=== DEPTH-SWEEP SQuAD EM/F1 SUMMARY -> $SUMMARY_LOG ==="
VAL_PATH="$VAL_PATH" "$PYBIN" - <<'PY' | tee "$SUMMARY_LOG"
import glob, json, os, subprocess, sys
rows = []
for sm in sorted(glob.glob("evidence_squad_label_prior/depthsweep_*/summary.json")):
    name = os.path.basename(os.path.dirname(sm))
    try:
        s = json.load(open(sm))
    except Exception as e:
        print(f"[warn] cannot read {sm}: {e}"); continue
    meta = s.get("meta") or {}
    rows.append((name, s.get("em"), s.get("f1"), s.get("n"),
                 meta.get("keep_front_layers"), meta.get("n_fresh_layers"),
                 os.path.basename(str(meta.get("val_path") or "?"))))
print(f"{'run':46s} {'keep':>4s} {'fresh':>5s} {'EM':>8s} {'F1':>8s} {'n':>6s}  val")
for name, em, f1, n, k, fr, vp in rows:
    em = em if em is not None else float('nan')
    f1 = f1 if f1 is not None else float('nan')
    print(f"{name:46s} {str(k):>4s} {str(fr):>5s} {em:8.4f} {f1:8.4f} {(n or 0):6d}  {vp}")
val = os.environ.get("VAL_PATH", "")
if val and os.path.exists(val):
    print("\n--- mandatory input-blind constant floor for this val set ---")
    sys.stdout.flush()
    subprocess.run([sys.executable, "scripts/report_constant_baseline.py", val])
PY
log "=== DEPTH-SWEEP DONE ==="
