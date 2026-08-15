#!/usr/bin/env bash
# ============================================================================
# UNION-9 GAP-FILL LAUNCHER — .212 (8x B200, sm_100, wzc1 disk)
#
# ⚠️⚠️ THE NUMBERS THIS SCRIPT PRODUCES ARE ONLY COMPARABLE TO ARMS MEASURED ON
#      sm_100 (compute_cap 10.0). Both completed token-matched arms were scored
#      on a compute_cap-10.0 box: their own lm_eval.log records
#      "max memory per GPU to {0: 190842863616, 1: ..., 2: ..., 3: ...}"
#      = 177.7 GiB x 4 GPUs, which is B200-class, not H20 (95 GB) and not a real
#      L20A (48 GB). `nvidia-smi` prints the name "NVIDIA L20A" on these boxes;
#      that string is a DISPLAY BUG (CLAUDE.md, memory/l20a-name-string-is-really-
#      b200-sm100.md). Judge the generation by compute_cap, never by the name.
#
#      Paper B measured a 0.03-0.16 pp cross-architecture floor on BIT-IDENTICAL
#      weights (status/PAPERB_CORE6_CROSSARCH_FLOOR.md). A gap-fill row scored on
#      H20 would therefore carry an architecture offset on top of the effect being
#      measured, which is exactly the cross-harness error class that already cost
#      this project a retraction (-0.346 pp AST-7, see
#      baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md). Hence the hard sm_100
#      guard below.
#
# ---------------------------------------------------------------------------
# WHY A GAP-FILL IS NEEDED — the matrix is ASYMMETRIC
# ---------------------------------------------------------------------------
# As of 2026-08-15 the token-matched +/-SLoRB comparison is CONFOUNDED, because
# the two completed arms differ in TWO things at once, not one:
#
#   arm      variant     zero_ratio      exact_2of4  union9_primary  ppl@4096
#   noslorb  hard_drop   0.500000000     1.0         59.5535         6.6795
#   slorb    hard_fold   0.0000000011    0.0         61.5413         6.1938
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                        TRUE 2:4 vs DENSE -- not an +/-SLoRB difference
#
# So the headline "+SLoRB wins 1.99 pp union-9" mixes the +/-SLoRB training
# decision with a sparse-vs-dense export difference. The slorb arm's own run log
# says so in as many words:
#   logs/sparseforge_tm_union9_slorb_progress.log:314
#     "** 2:4 COLUMN: BARRED. This arm's export is not exact 2:4 (SLoRB folded).
#        Report it, but never in a 2:4 column. **"
#
# HOW BIG IS THE CONFOUND? There is a direct measurement, on ONE checkpoint, so
# the variant is the only thing that moves (outputs/cast_eval_spec/sparseforge_5b/
# sparseforge_same_harness_table.json -> "headline"):
#   sparseforge_hard_drop  union9_primary 57.0678   ppl@4096 8.8290
#   sparseforge_hard_fold  union9_primary 62.4335   ppl@4096 6.2115
#   => variant ALONE = +5.37 pp union9 / -2.62 ppl
# Replicated on a second checkpoint (sparseforge_dolmino_link2/link2_summary.json,
# plain-acc union9): hard_drop 53.8748 -> hard_fold 58.9594 = +5.08 pp.
#
# 5.37 pp and 5.08 pp both DWARF the 1.99 pp being attributed to SLoRB. The
# cross-arm gap is therefore not interpretable as an +/-SLoRB effect at all until
# the variant is held fixed.
#
# ---------------------------------------------------------------------------
# WHICH CELL TO FILL — only ONE of the two is possible
# ---------------------------------------------------------------------------
# ARM=slorb VARIANT=hard_drop   <-- POSSIBLE, and it is the minimal sufficient cell
#     Verified 0-GPU on .212 (2026-08-15): applying nm_2_4_hard() to the slorb
#     ckpt's own masks and dropping the branch gives zero_frac=0.500000000 with
#     0 bad tiles on a 12-tensor sample (all 224 in-scope tensors present), so
#     export_sparseforge_to_hf.py:213's "mask=hard slorb=drop must yield exact
#     2:4" assertion PASSES. Pairs against the existing noslorb/hard_drop for a
#     variant-matched, 2:4-legal +/-SLoRB contrast.
#
# ARM=noslorb VARIANT=hard_fold  <-- IMPOSSIBLE, do not try
#     The noslorb ckpt contains 1411 tensors with SLoRB_Weight=0 and x_proj=0
#     (verified 0-GPU on .212). export_sparseforge_to_hf.py:181 hard-exits:
#     "--slorb fold requested but <...>.SLoRB_Weight/<...>.x_proj missing".
#     There is nothing to fold: the branch never existed. The script REFUSES
#     this combination rather than pretending otherwise.
#
# ⚠️ READ BEFORE INTERPRETING THE RESULT. slorb/hard_drop is a POST-HOC
#    AMPUTATION of a branch the model trained to depend on. That is Defect 1 of
#    baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md's CORRECTION block, which
#    RETRACTED a headline conclusion built on exactly this confusion, and it is
#    restated at _run_sparseforge_tokenmatched_union9_watcher.sh:56-74. So this
#    cell makes the 2:4 column variant-matched, but it does NOT by itself yield a
#    clean "+/-SLoRB at training time" number -- read the ledger note this script
#    writes before quoting it.
#
# ---------------------------------------------------------------------------
# MEASURED COST (from the two completed runs' own stage timestamps)
# ---------------------------------------------------------------------------
#   stage                       noslorb (01:09:58-01:19:13)  slorb (04:50:00-05:00:15)
#   1 export to HF                       2m10s                        3m03s
#   2 verify 2:4 PRE                       53s                          51s
#   3 PPL @4096 + @2048              22s + 20s                    25s + 20s
#   4 union-9 zero-shot (4 GPU)          4m53s                        4m56s
#   5 aggregate                             <1s                          <1s
#   6 verify 2:4 POST                      37s                          40s
#   TOTAL wall                       9m15s                       10m15s
# So ONE gap-fill cell is ~10 min wall on 4 GPUs (~0.7 GPU-h). Trivial.
#
# ---------------------------------------------------------------------------
# USAGE — DEFAULTS TO A DRY RUN. It will not touch a GPU unless you say so.
# ---------------------------------------------------------------------------
#   # preflight only (safe, 0 GPU):
#   bash scripts/launch_union9_gapfill_212.sh
#   # actually run it (MAIN decides when, after the audit concludes):
#   DRY_RUN=0 bash scripts/launch_union9_gapfill_212.sh
#   # explicit:
#   ARM=slorb VARIANT=hard_drop DRY_RUN=0 bash scripts/launch_union9_gapfill_212.sh
#
# Run this ON .212. Do NOT run it on LOCAL/.73/.82/.104 -- those 32 cards are
# carrying four 200k trainings (keep8/keep10/keep12/paperC).
# ============================================================================
set -u

ARM="${ARM:-slorb}"
VARIANT="${VARIANT:-hard_drop}"
DRY_RUN="${DRY_RUN:-1}"          # ★ default 1: preflight, then STOP.
GPUS="${GPUS:-0,1,2,3}"          # 4 GPUs -- matches both completed arms exactly.
REQUIRE_SM="${REQUIRE_SM:-10.0}"
SKIP_ARCH_GUARD="${SKIP_ARCH_GUARD:-0}"

ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
MOM="$ROOT/Mixture-of-Memory"
# venv_union9 lives ON THE wzc1 PROJECT DISK, so it survives node restarts and is
# visible from LOCAL and .212 alike (memory/persist-artifacts-on-wzc1-or-diskb.md).
# Its pyvenv.cfg has include-system-site-packages=true over
# /opt/conda/envs/torch-base, i.e. conda's torch 2.13.0 plus lm_eval 0.4.8 +
# transformers 4.57.6 pinned inside the venv. The bare conda env has NO lm_eval.
PY="${PYTHON_BIN:-$ROOT/venv_union9/bin/python}"
LM_EVAL="${LM_EVAL_BIN:-$ROOT/venv_union9/bin/lm_eval}"

TOOLS=$MOM/baselines/cast_repro/tools
EXPORT=$TOOLS/export_sparseforge_to_hf.py
VERIFY=$TOOLS/verify_2of4_hf_export.py
AGG9=$TOOLS/aggregate_zeroshot_union9.py
DATAPROBE=$TOOLS/probe_union9_datasets.py
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
REF_MODEL=$ROOT/models/Llama--Llama2-7b

# Byte-identical to _run_sparseforge_tokenmatched_union9_watcher.sh:202.
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

# boolq + rte resolve through hub parquet redirects, so the proxy is REQUIRED.
export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'

GPU0="${GPUS%%,*}"

ts ()  { date '+%F %T'; }
log () { echo "[$(ts)] $*"; }
die () { echo "[$(ts)] FATAL: $*" >&2; exit 2; }

# --------------------------------------------------------------- ARM/VARIANT MAP
# The (arm, variant) pair determines the export flags. It is asserted, never
# hand-picked, because the wrong pairing silently produces a number that answers
# a different question (see the header's amputation note).
case "$ARM" in
  slorb)
    CK_DIR=$ROOT/out_llama_tokenmatched_slorb/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260814_125037
    EXPECT_SLORB_TENSORS=1
    EXPECT_CK_BYTES=41078402630
    ;;
  noslorb)
    CK_DIR=$ROOT/out_llama_tokenmatched_noslorb/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260814_124534
    EXPECT_SLORB_TENSORS=0
    EXPECT_CK_BYTES=39381385794
    ;;
  *) die "unknown ARM=$ARM (expected slorb|noslorb)" ;;
esac

case "$VARIANT" in
  hard_drop) MASK=hard; SLORB=drop ;;
  hard_fold) MASK=hard; SLORB=fold ;;
  soft_fold) MASK=soft; SLORB=fold ;;
  *) die "unknown VARIANT=$VARIANT (expected hard_drop|hard_fold|soft_fold)" ;;
esac

# Refuse the impossible cell up front rather than burning an export.
if [ "$EXPECT_SLORB_TENSORS" -eq 0 ] && [ "$SLORB" = "fold" ]; then
  die "ARM=noslorb VARIANT=$VARIANT is IMPOSSIBLE. That ckpt has SLoRB_Weight=0 / x_proj=0
      (verified 0-GPU 2026-08-15), so export_sparseforge_to_hf.py:181 hard-exits
      '--slorb fold requested but ...SLoRB_Weight/...x_proj missing'. There is no branch
      to fold -- the model never had one. Use ARM=slorb VARIANT=hard_drop instead."
fi

# Refuse to silently redo a cell that already exists.
ALREADY=$ROOT/outputs/cast_eval_spec/sparseforge_tokenmatched_$ARM/$VARIANT/zeroshot_union9.json
if [ -f "$ALREADY" ]; then
  die "$ALREADY already exists -- this cell is already measured. Refusing to overwrite a
      recorded measurement. Move it aside first if a re-measure is genuinely intended."
fi

CK=$CK_DIR/model.pt
OUT=$ROOT/outputs/cast_eval_spec/sparseforge_tokenmatched_$ARM
EXPDIR=$ROOT/outputs/sparseforge_tokenmatched_${ARM}_hf
PROG=$MOM/logs/union9_gapfill_${ARM}_${VARIANT}.log

mkdir -p "$MOM/logs" "$OUT" "$EXPDIR"

log "======== UNION-9 GAP-FILL  ARM=$ARM VARIANT=$VARIANT (mask=$MASK slorb=$SLORB) ========" | tee -a "$PROG"
log "ROOT=$ROOT  PY=$PY  LM_EVAL=$LM_EVAL"                | tee -a "$PROG"
log "CK=$CK"                                              | tee -a "$PROG"
log "GPUS=$GPUS  DRY_RUN=$DRY_RUN  REQUIRE_SM=$REQUIRE_SM"| tee -a "$PROG"

# ==========================================================================
# PREFLIGHT — every check fatal, ALL of it before any GPU work.
# ==========================================================================
log "---- PREFLIGHT ----" | tee -a "$PROG"

# P0. assets
for f in "$EXPORT" "$VERIFY" "$AGG9" "$DATAPROBE" "$HARNESS_PPL" "$WIKI" \
         "$REF_MODEL/model.safetensors.index.json" "$REF_MODEL/tokenizer.model" "$CK"; do
  [ -e "$f" ] || die "missing asset: $f (two-disk trap: this launcher is wzc1-only; scp -O if ever moved)"
done
[ -x "$PY" ]       || die "PYTHON_BIN not executable: $PY"
[ -x "$LM_EVAL" ]  || die "lm_eval binary not found/executable: $LM_EVAL
      NOTE: the bare /opt/conda/envs/torch-base has NO lm_eval. The pinned stack is the
      project-disk venv $ROOT/venv_union9. Do not substitute another interpreter."
log "P0 assets OK" | tee -a "$PROG"

# P1. checkpoint byte-size identity. The two completed arms were scored from these
#     exact files; a different size means a different (or truncated) ckpt. A
#     truncated model.pt is a real failure mode in this repo -- out_llama/
#     ..._20260306_211245/model.pt is 6.7 GB against a 47.7 GB sibling and raises
#     PytorchStreamReader "failed finding central directory".
CK_BYTES=$(stat -c %s "$CK") || die "cannot stat $CK"
[ "$CK_BYTES" = "$EXPECT_CK_BYTES" ] \
  || die "ckpt size $CK_BYTES != expected $EXPECT_CK_BYTES for ARM=$ARM. Wrong or truncated ckpt."
log "P1 ckpt bytes=$CK_BYTES (matches the arm scored on 2026-08-15)" | tee -a "$PROG"

# P2. HARNESS ASSERTION. lm_eval MUST be 0.4.8 and transformers 4.57.6. Every arm
#     in the union-9 table records "transformers_version": "4.57.6" in its
#     results_*.json. A stack drift here is the retraction-class error.
#     ⚠️ lm_eval 0.4.8 does NOT define lm_eval.__version__ -- use the distribution
#     metadata (same source as the watcher's assertion at line 253).
CUDA_VISIBLE_DEVICES="" "$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import importlib.metadata as m
want = {"lm_eval": "0.4.8", "transformers": "4.57.6"}
bad = []
for pkg, exp in want.items():
    try:
        got = m.version(pkg)
    except Exception as e:
        got = f"<missing: {e.__class__.__name__}>"
    print(f"[harness] {'OK ' if got == exp else 'DRIFT'} {pkg}: got {got} expected {exp}")
    if got != exp:
        bad.append(pkg)
print("[harness] VERDICT: " + ("MATCH" if not bad else f"MISMATCH on {bad}"))
raise SystemExit(0 if not bad else 21)
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] \
  || die "harness stack != lm_eval 0.4.8 / transformers 4.57.6. Refusing to add a row measured
      on a different stack -- that is the -0.346 pp cross-harness error that already forced a
      retraction. Fix the env; do NOT set an override."

# P3. torch version. 2.7.0 vs 2.13.0 alone moved ~20 items on bit-identical
#     weights (status/PAPERB_FLIP_BOUNDARY_RESOLVED.md).
TORCH_VER=$(CUDA_VISIBLE_DEVICES="" "$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null) \
  || die "cannot import torch with $PY"
log "P3 torch=$TORCH_VER" | tee -a "$PROG"
case "$TORCH_VER" in
  2.13.*) : ;;
  *) die "torch=$TORCH_VER but the union-9 arms were all measured on torch 2.13.x." ;;
esac

# P4. ARCHITECTURE GUARD. See the header: judge by compute_cap, NOT by the name
#     string ("NVIDIA L20A" is a display bug on these B200 boxes).
CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' '/')
log "P4 compute_cap(s)=[${CAPS%,}]  name(s)=[${GPU_NAME%/}]  (name is NOT authoritative)" | tee -a "$PROG"
if [ "$SKIP_ARCH_GUARD" != "1" ]; then
  case "$CAPS" in
    "${REQUIRE_SM},") : ;;
    *) die "this node reports compute_cap=[${CAPS%,}] but the union-9 gap-fill requires
      ${REQUIRE_SM} (sm_100 / B200, i.e. LOCAL or .212). Both completed token-matched arms were
      scored on a cc-10.0 box (their lm_eval.log shows 190842863616 B = 177.7 GiB per GPU).
      Paper B measured a 0.03-0.16 pp cross-arch floor on BIT-IDENTICAL weights, so an H20 row
      would carry an architecture offset on top of the effect. Run on .212." ;;
  esac
else
  log "P4 WARNING: SKIP_ARCH_GUARD=1 -- this row will NOT be architecture-comparable to the
      existing token-matched arms. You must record why, in writing, next to the number." | tee -a "$PROG"
fi

# P5. GPU count must match the completed arms (both used exactly 4).
N_GPU_REQ=$(printf '%s' "$GPUS" | awk -F, '{print NF}')
[ "$N_GPU_REQ" -eq 4 ] \
  || die "GPUS=$GPUS is $N_GPU_REQ GPUs, but both completed token-matched arms ran on exactly 4
      (parallelize=True, 'max memory per GPU to {0:,1:,2:,3:}'). Changing the shard topology
      changes the auto-batch-size search. Keep 4."

# P6. my GPUs must be free. Do NOT compete for cards.
USED=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk '{s+=$1} END {print s+0}')
log "P6 memory currently held on GPUs $GPUS = ${USED}MiB" | tee -a "$PROG"
[ "${USED:-999999}" -lt 8000 ] || die "${USED}MiB already held on GPUs $GPUS. Refusing to compete
      for cards. Identify the owner with: nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv"

# P7. proxy reachability -- boolq/rte need the hub.
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
       https://huggingface.co/api/datasets/aps/super_glue 2>/dev/null || echo 000)
log "P7 hub probe HTTP $CODE" | tee -a "$PROG"
[ "$CODE" = "200" ] || log "P7 WARNING: hub returned $CODE; boolq/rte may not resolve. Watch STAGE 4." | tee -a "$PROG"

# P8. ★ COMPLETENESS PRE-CHECK: all 9 tasks must LOAD, each with n identical to the
#     completed arms. This is the "n_scored == expected" discipline, not just a NaN
#     check: a 6/8-shard silent merge once produced a complete-looking artefact with
#     n_nan=0 on a changed measurement basis
#     (memory/same-harness-runs-bit-identical.md, kill-remote-gpu-job-by-pid-not-pkill.md).
#     ⚠️ piqa only resolves because venv_union9 carries a task override
#     (venv_union9/lib/python3.14/site-packages/lm_eval/tasks/piqa/piqa.yaml);
#     upstream 0.4.8's piqa.yaml uses a loading SCRIPT that datasets 5.0.1 refuses.
log "P8 probing all 9 task datasets (0 GPU)..." | tee -a "$PROG"
CUDA_VISIBLE_DEVICES="" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
  "$PY" "$DATAPROBE" 2>&1 | tee -a "$PROG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "dataset preflight FAILED -- at least one of the 9 tasks
      does not load, or loads a different number of docs than the completed arms. Launching now
      would produce a well-formed results json on a DIFFERENT measurement basis. Fix first."

# P9. piqa override identity (hash-level), since P8 only checks the count.
if [ -f "$TOOLS/verify_piqa_override_hashes.py" ]; then
  log "P9 verifying piqa override is hash-identical to the archive..." | tee -a "$PROG"
  CUDA_VISIBLE_DEVICES="" "$PY" "$TOOLS/verify_piqa_override_hashes.py" 2>&1 | tail -6 | tee -a "$PROG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "piqa override is NOT hash-identical to the archive. The piqa
      cell would not be comparable. Fix the override before scoring."
fi

log "---- PREFLIGHT PASSED ----" | tee -a "$PROG"

if [ "$DRY_RUN" != "0" ]; then
  log "DRY_RUN=$DRY_RUN -> stopping here. NOTHING was run on a GPU." | tee -a "$PROG"
  log "To execute: DRY_RUN=0 ARM=$ARM VARIANT=$VARIANT bash scripts/launch_union9_gapfill_212.sh" | tee -a "$PROG"
  exit 0
fi

# ==========================================================================
# STAGE 1 — export
# ==========================================================================
log "=== STAGE 1: export $VARIANT (mask=$MASK slorb=$SLORB) ===" | tee -a "$PROG"
if [ -f "$EXPDIR/$VARIANT/sparseforge_export_meta.json" ]; then
  log "--- $VARIANT already exported; reusing" | tee -a "$PROG"
else
  "$PY" "$EXPORT" --ckpt "$CK" --output "$EXPDIR/$VARIANT" \
      --mask "$MASK" --slorb "$SLORB" \
      --model "$REF_MODEL" --project-root "$ROOT" 2>&1 | tee "$OUT/export_$VARIANT.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "export failed -- a wrong export is worse than no number"
fi
log "=== STAGE 1 DONE ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 2 — 2:4 verify (PRE-inference)
# ==========================================================================
log "=== STAGE 2: verify 2:4 $VARIANT (PRE) ===" | tee -a "$PROG"
CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$VERIFY" \
    --model "$EXPDIR/$VARIANT" --sample-layers 12 --seed 0 2>&1 \
  | tee "$OUT/verify_2of4_${VARIANT}_pre.log"
pre_rc=${PIPESTATUS[0]}
log "STAGE 2 rc=$pre_rc" | tee -a "$PROG"

if [ "$SLORB" = "drop" ]; then
  # drop MUST be exact 2:4 -- export line 213 already refuses otherwise, so a FAIL
  # here is a REAL failure, never the by-design fold failure.
  [ "$pre_rc" -eq 0 ] || die "$VARIANT failed the 2:4 gate but slorb=drop must PASS. REAL failure
      (bad ckpt or export bug), not the by-design fold densification. Not scoring."
  ELIGIBLE_2OF4=true
  log "$VARIANT PASSES the 2:4 gate -> eligible for the 2:4 column" | tee -a "$PROG"
else
  if [ "$pre_rc" -eq 0 ]; then
    ELIGIBLE_2OF4=suspicious
    log "WARN $VARIANT PASSED the 2:4 gate; folding a live branch cannot leave the weight 2:4
      unless the branch is ~a no-op. Investigate before trusting." | tee -a "$PROG"
  else
    ELIGIBLE_2OF4=false
    log "$VARIANT FAILS the 2:4 gate -- EXPECTED (SLoRB fold densifies). Scored and reported,
      but BARRED from any 2:4 column." | tee -a "$PROG"
  fi
fi

# ==========================================================================
# STAGE 3 — WikiText-2 PPL at BOTH seqlens, each labelled
# ==========================================================================
log "=== STAGE 3: WikiText-2 PPL @4096 and @2048 ===" | tee -a "$PROG"
for SEQ in 4096 2048; do
  o=$OUT/$VARIANT/ppl${SEQ}
  mkdir -p "$o"
  log "--- ppl@${SEQ} $VARIANT" | tee -a "$PROG"
  CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$HARNESS_PPL" \
      --model "$EXPDIR/$VARIANT" --output_dir "$o" --wiki_text "$WIKI" \
      --seqlen "$SEQ" --wiki_tokens 100000000 --device cuda:0 2>&1 | tee "$o/ppl${SEQ}.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "PPL@${SEQ} failed -- a row missing a PPL cell is how the
      2048-vs-4096 mixup happened in the first place. Not continuing."
done
log "=== STAGE 3 DONE ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 4 — union-9 zero-shot. Invocation byte-identical to the completed arms
#           except `pretrained`. --batch_size auto is deliberate: the existing
#           rows record batch_size="auto" resolving to [64]. Hard-coding 64 would
#           be a DIFFERENT invocation string.
# ==========================================================================
log "=== STAGE 4: union-9 zero-shot on GPUs $GPUS ===" | tee -a "$PROG"
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
[ "${PIPESTATUS[0]}" -eq 0 ] || die "lm_eval failed -- not aggregating a partial run"
log "=== STAGE 4 DONE ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 5 — aggregate. agg9 hard-fails if ANY of the 9 is absent (its line 103).
# ==========================================================================
log "=== STAGE 5: aggregate (asserts all 9 tasks present) ===" | tee -a "$PROG"
ITER=$(CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch,sys
print(torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True).get('iter_num','unk'))
" "$CK" 2>/dev/null || echo unk)
"$PY" "$AGG9" \
    --lm-eval-out "$o/lm_eval_out" \
    --output "$o/zeroshot_union9.json" \
    --model "sparseforge_tokenmatched_${ARM}_iter${ITER}_${VARIANT}" 2>&1 | tee -a "$PROG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "aggregation failed -- row INVALID, not writing a summary"
log "=== STAGE 5 DONE ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 6 — 2:4 verify (POST-inference)
# ==========================================================================
log "=== STAGE 6: verify 2:4 $VARIANT (POST) ===" | tee -a "$PROG"
CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$VERIFY" \
    --model "$EXPDIR/$VARIANT" --sample-layers 12 --seed 0 2>&1 \
  | tee "$OUT/verify_2of4_${VARIANT}_post.log"
post_rc=${PIPESTATUS[0]}
log "=== STAGE 6 DONE rc=$post_rc ===" | tee -a "$PROG"
[ "$pre_rc" -eq "$post_rc" ] \
  || log "WARN 2:4 gate changed across inference (pre=$pre_rc post=$post_rc) -- investigate" | tee -a "$PROG"

# ==========================================================================
# SUMMARY — ★ this is where the n_scored == expected assertion is ENFORCED,
#           per-task, against the arms already in the table.
# ==========================================================================
log "=== SUMMARY ===" | tee -a "$PROG"
CUDA_VISIBLE_DEVICES="" "$PY" - "$ARM" "$VARIANT" "$CK" "$ITER" "$OUT" "$ELIGIBLE_2OF4" \
                              "$pre_rc" "$post_rc" "$CAPS" "$GPUS" <<'PYEOF' 2>&1 | tee -a "$PROG"
import json, sys, pathlib

arm, variant, ck, it, out, elig, pre_rc, post_rc, caps, gpus = sys.argv[1:11]
out = pathlib.Path(out)

# n_samples recorded by BOTH completed token-matched arms, and identically by every
# arm in sparseforge_5b/sparseforge_same_harness_table.json. A gap-fill cell that
# scores a different number of docs is NOT comparable, however clean its json looks.
EXPECT_N = {"boolq": 3270, "rte": 277, "hellaswag": 10042, "race": 1045, "piqa": 1838,
            "winogrande": 1267, "arc_easy": 2376, "arc_challenge": 1172, "openbookqa": 500}
# Primary metric per task, fixed by the table (agg9's PRIMARY_METRIC map).
EXPECT_PRIMARY = {"boolq": "acc", "rte": "acc", "hellaswag": "acc_norm", "race": "acc",
                  "piqa": "acc", "winogrande": "acc", "arc_easy": "acc_norm",
                  "arc_challenge": "acc_norm", "openbookqa": "acc_norm"}

S = {"arm": arm, "variant": variant, "source_ckpt": ck, "source_iter": it,
     "2of4_eligible": elig, "verify_pre_rc": int(pre_rc), "verify_post_rc": int(post_rc),
     "compute_caps": caps.rstrip(","), "gpus": gpus,
     "tasks": ",".join(EXPECT_N), "n_tasks": 9,
     "harness": "lm_eval 0.4.8, --model hf, dtype=bfloat16, parallelize=True, "
                "add_bos_token=False, --batch_size auto, --num_fewshot 0, --seed 0",
     "comparable_only_to": "arms measured on compute_cap 10.0 (sm_100/B200)",
     "ppl": {}}

for seq in ("4096", "2048"):
    f = out / variant / f"ppl{seq}" / "ppl_metrics.json"
    if not f.exists():
        raise SystemExit(f"FATAL: missing {f} -- refusing to write a summary without both PPLs")
    d = json.load(open(f))
    assert str(d["seqlen"]) == seq, f"seqlen mislabel: file says {d['seqlen']} in ppl{seq}/"
    S["ppl"][seq] = {"wikitext2_ppl": d["wikitext2_ppl"],
                     "wikitext2_tokens": d["wikitext2_tokens"],
                     "linear_zero_ratio": d["linear_zero_ratio"],
                     "exact_2of4_tile_ratio": d["exact_2of4_tile_ratio"]}
    print(f"  PPL @{seq}: {d['wikitext2_ppl']:.6f}  tokens={d['wikitext2_tokens']} "
          f"zero_ratio={d['linear_zero_ratio']:.9f} exact_2of4={d['exact_2of4_tile_ratio']}")

z = out / variant / "zeroshot_union9.json"
if not z.exists():
    raise SystemExit(f"FATAL: {z} absent")
b = json.load(open(z))
per = b["per_task"]

# ---- COMPLETENESS ASSERTIONS -------------------------------------------------
# (1) all 9 present; (2) per-task n EXACTLY equal to the completed arms;
# (3) primary metric per task unchanged; (4) the primary value is not None.
missing = sorted(set(EXPECT_N) - set(per))
if missing:
    raise SystemExit(f"FATAL: missing tasks {missing} -- row INVALID")
extra = sorted(set(per) - set(EXPECT_N))
if extra:
    raise SystemExit(f"FATAL: unexpected extra tasks {extra} -- row INVALID")
bad = []
for t, want in EXPECT_N.items():
    e = per[t]
    got = e.get("n_samples")
    if got != want:
        bad.append(f"{t}: n_scored={got} != expected={want}")
    if e.get("primary_metric") != EXPECT_PRIMARY[t]:
        bad.append(f"{t}: primary_metric={e.get('primary_metric')} != {EXPECT_PRIMARY[t]}")
    if e.get(EXPECT_PRIMARY[t]) is None:
        bad.append(f"{t}: primary value is None")
if bad:
    raise SystemExit("FATAL n_scored/metric mismatch vs the completed arms:\n  "
                     + "\n  ".join(bad)
                     + "\n  A results json with n_nan=0 can still be on a DIFFERENT measurement "
                       "basis (cf. the 6/8-shard silent merge). Row INVALID.")
print(f"  COMPLETENESS OK: 9/9 tasks, per-task n_scored identical to the completed arms")
S["per_task"] = {t: {"acc": e["acc"], "acc_norm": e["acc_norm"],
                     "primary_metric": e["primary_metric"], "n_samples": e["n_samples"]}
                 for t, e in per.items()}

for k in ("union9", "cast7", "ast7"):
    s = b[k]
    S[k] = {"mean_primary": s["mean_primary"], "mean_plain_acc": s["mean_plain_acc"]}
    print(f"  {k:7}: plain_acc {s['mean_plain_acc']*100:.4f}  primary {s['mean_primary']*100:.4f}")

r = per["rte"]
k_int = r["acc"] * r["n_samples"]
S["rte_integral"] = {"acc": r["acc"], "n": r["n_samples"], "k": k_int, "k_rounded": round(k_int)}
print(f"  RTE: acc={r['acc']:.10f} n={r['n_samples']} k={k_int:.4f} -> {round(k_int)}/{r['n_samples']}")
assert abs(k_int - round(k_int)) < 0.01, f"RTE k={k_int} not integral -- metric mismatch"
assert r["n_samples"] == 277, f"RTE n={r['n_samples']} != 277 -- wrong split"

dest = out / f"gapfill_union9_summary_{variant}.json"
dest.write_text(json.dumps(S, indent=2) + "\n")
print(f"  wrote {dest}")

if elig != "true":
    print("  ** 2:4 COLUMN: BARRED (export is not exact 2:4). Report it, never in a 2:4 column. **")
print("  ** INTERPRETATION GUARD: if arm=slorb and variant=hard_drop, this cell is a POST-HOC")
print("     AMPUTATION of a branch the model trained to depend on (SPARSEFORGE_SAME_HARNESS.md")
print("     CORRECTION, Defect 1). It makes the 2:4 column variant-matched; it is NOT by itself")
print("     a clean 'training-time +/-SLoRB' number. Do not quote it as one. **")
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || die "summary/completeness assertions FAILED -- row INVALID"

log "=== ALL STAGES COMPLETE (ARM=$ARM VARIANT=$VARIANT) ===" | tee -a "$PROG"
