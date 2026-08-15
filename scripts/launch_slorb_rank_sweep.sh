#!/usr/bin/env bash
# =============================================================================
# B12 — SLoRB rank-ladder sweep driver.  *** DRY_RUN=1 BY DEFAULT ***
#
# THIS SCRIPT HAS NEVER BEEN EXECUTED, IN ANY MODE.  It was written 2026-08-16
# alongside proposal/backlog/B12-slorb-rank-efficiency/PROPOSAL.md and was not
# run.  Do not claim any number from it until it has been.
#
# WHAT IT DOES
#   For one frozen ladder rung: emit the variant (CPU) -> verify -> ppl -> union-9
#   zero-shot -> aggregate -> assert per-task n_scored == expected -> write a row.
#   Structure and every preflight gate mirror scripts/launch_union9_gapfill_212.sh,
#   which is the driver that produced the cell this ladder is anchored to.
#
# GPU BUDGET — READ THIS FIRST
#   Authorised spend at the time of writing: NOTHING.  The proposal's next_gate is
#   0-GPU (novelty write-up + the CPU rung-A export self-check).  When that passes,
#   the authorised spend is the PILOT PAIR ONLY: rung P then rung Dctl, 2 x 0.73 =
#   1.46 GPU-h.  Rungs Q/R/S are NOT authorised and need the pilot verdict first.
#
# WHY EACH GUARD IS HERE (all four were requested explicitly, all four mirror the
# gap-fill driver, and each one corresponds to a real error already made here):
#   P4  compute_cap MUST be 10.0.  Judge the generation by compute_cap, NEVER by
#       the name: nvidia-smi prints "NVIDIA L20A" on these boxes and they are
#       really B200/sm_100 (memory/l20a-name-string-is-really-b200-sm100.md).
#       Scoring on H20 sm_90 injects the measured 0.03-0.16 pp cross-arch offset
#       on top of an effect whose whole threshold is 1.79 pp.
#   P2  lm_eval version via importlib.metadata, NOT lm_eval.__version__ --
#       lm_eval 0.4.8 does not define __version__ at all, so an attribute read
#       would raise and a `getattr(..., 'unknown')` would silently pass.
#   P6  refuse to run if ANY GPU on the node holds memory.  Not just the 4 we
#       want: this node's other cards may be mid-training, and an OOM caused by
#       our arrival has already destroyed 4/5 rungs of someone else's eval
#       (memory/subagent-prompt-must-state-gpu-budget.md).
#   S5  per-task n_scored == expected, NOT a NaN check.  A silent 6/8-shard merge
#       once produced a complete-looking artefact with n_nan=0 on a CHANGED
#       measurement basis (memory/same-harness-runs-bit-identical.md).
#
# USAGE
#   # 0-GPU self-check (what the next_gate actually asks for):
#   RUNG=A DRY_RUN=1 bash scripts/launch_slorb_rank_sweep.sh
#   # pilot, only after the gate passes and only with authorisation:
#   RUNG=P DRY_RUN=0 bash scripts/launch_slorb_rank_sweep.sh
# =============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
REPO="${REPO:-$PROJECT_ROOT/Mixture-of-Memory}"
TOOLS="$REPO/baselines/cast_repro/tools"

RUNG="${RUNG:-A}"
DRY_RUN="${DRY_RUN:-1}"                 # ★ default 1. Never flip this in the file.
GPUS="${GPUS:-0,1,2,3}"                 # both completed arms used exactly 4
GPU0="${GPU0:-0}"
REQUIRE_SM="${REQUIRE_SM:-10.0}"        # sm_100 / B200. Mirrors the gap-fill driver.
COEFFS="${COEFFS:-ls}"                  # measured 4.22% better than naive, same cost

# The 5B headline checkpoint. wzc1-ONLY (zwfy6 has no outputs/cast_eval_spec at all).
CK="${CK:-$PROJECT_ROOT/out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt}"
EXPECT_CK_BYTES="${EXPECT_CK_BYTES:-41078444091}"

OUT="${OUT:-$PROJECT_ROOT/outputs/cast_eval_spec/slorb_rank_ladder}"
EXPDIR="${EXPDIR:-$PROJECT_ROOT/outputs/slorb_rank_ladder_hf}"
PROG="$REPO/logs/slorb_rank_sweep_${RUNG}.log"

# venv_union9 lives under PROJECT_ROOT, one level ABOVE the checkout -- not under $REPO.
# Measured 2026-08-16: $REPO/venv_union9/bin/python does not exist, $PROJECT_ROOT/venv_union9/bin/python
# does and carries the union-9 harness (lm_eval 0.4.8 / transformers 4.57.6). With the old $REPO paths the
# `[ -x "$PY" ]` guard below fired immediately, so this driver had never been runnable as written.
PY="${PY:-$PROJECT_ROOT/venv_union9/bin/python}"
LM_EVAL="${LM_EVAL:-$PROJECT_ROOT/venv_union9/bin/lm_eval}"
EMIT="$TOOLS/emit_slorb_ladder.py"
AGG9="$TOOLS/aggregate_zeroshot_union9.py"
VERIFY="$TOOLS/../verify_2of4_hf_export.py"
HARNESS_PPL="${HARNESS_PPL:-$REPO/baselines/eval_hf_sparse_model.py}"
DATAPROBE="$TOOLS/probe_union9_datasets.py"
WIKI="${WIKI:-$PROJECT_ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw}"
TASKS="boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa"

mkdir -p "$(dirname "$PROG")" "$OUT" "$EXPDIR"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "FATAL: $*" | tee -a "$PROG" >&2; exit 1; }

log "==== B12 SLoRB rank ladder: RUNG=$RUNG DRY_RUN=$DRY_RUN GPUS=$GPUS REQUIRE_SM=$REQUIRE_SM ====" | tee -a "$PROG"

# ---------------------------------------------------------------------------
# The frozen ladder, duplicated here ONLY to fail fast on a bad RUNG name and to
# print the pre-registered expectation. emit_slorb_ladder.py owns the real map and
# asserts against its own copy; if the two ever disagree the emit tool hard-exits.
# ---------------------------------------------------------------------------
case "$RUNG" in
  A)    EXP_DENS=56.2500; EXP_LIVE=404750336; EXP_PSI=0.5229; QUALIFIES=no  ;;
  P)    EXP_DENS=55.0316; EXP_LIVE=325844992; EXP_PSI=0.6159; QUALIFIES=yes ;;
  Q)    EXP_DENS=54.4912; EXP_LIVE=290848768; EXP_PSI=0.6572; QUALIFIES=yes ;;
  R)    EXP_DENS=54.1026; EXP_LIVE=265682944; EXP_PSI=0.6869; QUALIFIES=yes ;;
  S)    EXP_DENS=52.9287; EXP_LIVE=189661184; EXP_PSI=0.7765; QUALIFIES=yes ;;
  Dctl) EXP_DENS=55.0355; EXP_LIVE=326098944; EXP_PSI=0.6156; QUALIFIES=control ;;
  R0|R1) die "rung $RUNG is an ANCHOR and is ALREADY ON DISK (R0 = sparseforge_hard_fold
      union-9 62.4335 at density 63.1011%; R1 = sparseforge_hard_drop union-9 57.0678 at
      50.0000%). Re-scoring it would spend 0.73 GPU-h to reproduce a number we have, and if
      it came out different that would be a harness-drift finding, not a ladder rung." ;;
  *) die "unknown RUNG=$RUNG. Frozen ladder is A,P,Q,R,S,Dctl (+ on-disk anchors R0,R1)." ;;
esac
log "rung $RUNG: expected density ${EXP_DENS}% branch_live ${EXP_LIVE} psi ${EXP_PSI} qualifies=${QUALIFIES}" | tee -a "$PROG"
if [ "$RUNG" != "A" ] && [ "$RUNG" != "P" ] && [ "$RUNG" != "Dctl" ]; then
  log "NOTE: rung $RUNG is NOT authorised by PROPOSAL.md -- only the pilot pair (P, Dctl)
      is, and only after the 0-GPU gate. Proceeding requires an explicit authorisation
      recorded in STATUS.json." | tee -a "$PROG"
fi

# ==========================================================================
# PREFLIGHT
# ==========================================================================
# P0. tools present.
for f in "$EMIT" "$AGG9" "$HARNESS_PPL" "$DATAPROBE"; do
  [ -f "$f" ] || die "missing required tool: $f"
done
[ -x "$PY" ] || die "python not executable: $PY"

# P1. checkpoint identity by BYTE SIZE. A truncated or wrong ckpt is the one error
#     that would silently change every rung at once.
[ -f "$CK" ] || die "checkpoint not found: $CK  (this asset is wzc1-ONLY; zwfy6 has no
      outputs/cast_eval_spec and no copy of this file)"
CK_BYTES=$(stat -c %s "$CK")
[ "$CK_BYTES" = "$EXPECT_CK_BYTES" ] \
  || die "ckpt size $CK_BYTES != expected $EXPECT_CK_BYTES. Wrong or truncated checkpoint;
      every rung in the ladder is defined against this exact file."
log "P1 ckpt bytes=$CK_BYTES OK" | tee -a "$PROG"

# P2. HARNESS ASSERTION via importlib.metadata.
#     ⚠️ lm_eval 0.4.8 has NO __version__ attribute -- reading it raises, and a
#     getattr fallback would silently accept any version. Distribution metadata is
#     the only reliable source, and it is what the gap-fill driver uses.
CUDA_VISIBLE_DEVICES="" "$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import importlib.metadata as m
want = {"lm_eval": "0.4.8", "transformers": "4.57.6"}
bad = []
for pkg, exp in want.items():
    try:
        got = m.version(pkg)
    except Exception as e:
        got = f"<missing: {e.__class__.__name__}>"
    print(f"[harness] {'OK   ' if got == exp else 'DRIFT'} {pkg}: got {got} expected {exp}")
    if got != exp:
        bad.append(pkg)
# Demonstrate WHY metadata is used, so nobody 'simplifies' this back to __version__.
try:
    import lm_eval
    print(f"[harness] lm_eval.__version__ present? "
          f"{hasattr(lm_eval, '__version__')} (0.4.8 -> False; never read it)")
except Exception as e:
    print(f"[harness] note: import lm_eval failed ({e.__class__.__name__})")
print("[harness] VERDICT: " + ("MATCH" if not bad else f"MISMATCH on {bad}"))
raise SystemExit(0 if not bad else 21)
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] \
  || die "harness stack != lm_eval 0.4.8 / transformers 4.57.6. Refusing to add a row measured
      on a different stack -- cross-harness drift already forced a retraction here (-0.346 pp
      AST-7). Fix the env; do NOT add an override flag."

# P3. torch version. 2.7.0 vs 2.13.0 alone moved ~20 items on BIT-IDENTICAL weights.
TORCH_VER=$(CUDA_VISIBLE_DEVICES="" "$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null) \
  || die "cannot import torch with $PY"
log "P3 torch=$TORCH_VER" | tee -a "$PROG"
case "$TORCH_VER" in
  2.13.*) : ;;
  *) die "torch=$TORCH_VER but every union-9 arm in the 5B table was measured on torch 2.13.x." ;;
esac

# P4. ARCHITECTURE GUARD. compute_cap, never the name string.
CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' '/')
log "P4 compute_cap(s)=[${CAPS%,}] name(s)=[${GPU_NAME%/}] (the NAME IS NOT AUTHORITATIVE:
      these boxes report 'NVIDIA L20A' and are really B200/sm_100)" | tee -a "$PROG"
[ -n "$CAPS" ] || die "nvidia-smi returned no compute_cap -- cannot verify the architecture,
      and an unverified arch is exactly the confound REQUIRE_SM exists to exclude."
case "$CAPS" in
  "${REQUIRE_SM},") : ;;
  *) die "this node reports compute_cap=[${CAPS%,}] but the B12 ladder requires ${REQUIRE_SM}
      (sm_100 / B200, i.e. LOCAL or .212). Every row of the 5B same-harness table was scored on
      cc 10.0. An H20 sm_90 row would carry the measured 0.03-0.16 pp cross-arch offset, against
      a decision threshold of tau=1.79 pp. Also note the checkpoint is wzc1-only, so the H20
      nodes could not read it anyway." ;;
esac

# P5. GPU count must match the completed arms (both used exactly 4) -- the shard
#     topology changes the auto-batch-size search, hence the invocation string.
N_GPU_REQ=$(printf '%s' "$GPUS" | awk -F, '{print NF}')
[ "$N_GPU_REQ" -eq 4 ] \
  || die "GPUS=$GPUS is $N_GPU_REQ GPUs, but every completed cell ran on exactly 4."

# P6. ★ NO GPU ON THIS NODE MAY BE BUSY. Deliberately stricter than the gap-fill
#     driver, which only checked the 4 cards it wanted. Checking all of them is the
#     lesson from destroying 4/5 rungs of a co-tenant's eval by inducing an OOM.
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk -F', *' '$2 > 1000 {printf "%s(%sMiB) ", $1, $2}')
TOTUSED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk '{s+=$1} END {print s+0}')
log "P6 node-wide memory held = ${TOTUSED}MiB; busy cards: ${BUSY:-none}" | tee -a "$PROG"
if [ -n "$BUSY" ]; then
  die "GPUs already busy on this node: $BUSY
      Refusing to run. This is not only about competing for the 4 cards we want -- arriving on
      a node whose other cards are mid-training can OOM the co-tenant, which has already cost
      4 of 5 rungs of another eval here. Identify the owner with:
        nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
      and coordinate before launching."
fi

# P7. hub reachability -- boolq/rte need it.
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
       https://huggingface.co/api/datasets/aps/super_glue 2>/dev/null || echo 000)
log "P7 hub probe HTTP $CODE" | tee -a "$PROG"
[ "$CODE" = "200" ] || log "P7 WARNING: hub returned $CODE; boolq/rte may not resolve." | tee -a "$PROG"

# P8. COMPLETENESS PRE-CHECK: all 9 tasks must LOAD with n identical to the table.
log "P8 probing all 9 task datasets (0 GPU)..." | tee -a "$PROG"
CUDA_VISIBLE_DEVICES="" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
  "$PY" "$DATAPROBE" 2>&1 | tee -a "$PROG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "dataset preflight FAILED -- a task does not load, or loads a
      different number of docs than the completed arms. Launching would produce a well-formed
      results json on a DIFFERENT measurement basis."

# P9. ★ 0-GPU EMIT SELF-CHECK. Runs the emit tool in --dry-run on THIS rung and
#     lets it assert its own density/param bookkeeping against the pre-registered
#     constants. For rung A the answer is independently known (56.2500% /
#     404,750,336), which is why rung A is the prescribed first execution ever.
log "P9 emit --dry-run bookkeeping assertion for rung $RUNG (0 GPU)..." | tee -a "$PROG"
CUDA_VISIBLE_DEVICES="" "$PY" "$EMIT" --ckpt "$CK" --rung "$RUNG" --coeffs "$COEFFS" \
    --output "$EXPDIR/rung_${RUNG}" --project-root "$PROJECT_ROOT" --dry-run 2>&1 | tee -a "$PROG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "emit --dry-run FAILED its own pre-registration assertion for
      rung $RUNG. Either the frozen ladder in STATUS.json or the code is wrong; resolve that
      BEFORE any GPU. (This is the check the proposal's next_gate leg 2 asks for.)"

log "---- PREFLIGHT PASSED ----" | tee -a "$PROG"

if [ "$DRY_RUN" != "0" ]; then
  log "DRY_RUN=$DRY_RUN -> stopping here. NOTHING was run on a GPU, no weights written." | tee -a "$PROG"
  log "To execute (requires authorisation per PROPOSAL.md): DRY_RUN=0 RUNG=$RUNG bash scripts/launch_slorb_rank_sweep.sh" | tee -a "$PROG"
  exit 0
fi

VAR="rung_${RUNG}"

# ==========================================================================
# STAGE 1 — emit the variant (CPU)
# ==========================================================================
log "=== STAGE 1: emit $VAR (coeffs=$COEFFS) ===" | tee -a "$PROG"
if [ -f "$EXPDIR/$VAR/slorb_ladder_manifest.json" ]; then
  log "--- $VAR already emitted; reusing" | tee -a "$PROG"
else
  CUDA_VISIBLE_DEVICES="" "$PY" "$EMIT" --ckpt "$CK" --rung "$RUNG" --coeffs "$COEFFS" \
      --output "$EXPDIR/$VAR" --project-root "$PROJECT_ROOT" --dtype bfloat16 2>&1 | tee -a "$PROG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "emit failed for rung $RUNG"
fi
log "=== STAGE 1 DONE ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 2 — 2:4 verify (PRE). Expected to FAIL: every rung is FOLDED and therefore
#           dense on disk. We record the rc, we do not require 2:4.
# ==========================================================================
log "=== STAGE 2: 2:4 verify $VAR (PRE; a FAIL is EXPECTED -- folded => dense) ===" | tee -a "$PROG"
pre_rc=0
if [ -f "$VERIFY" ]; then
  CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$VERIFY" --model "$EXPDIR/$VAR" \
      --sample-layers 12 --seed 0 2>&1 | tee "$OUT/verify_2of4_${VAR}_pre.log"
  pre_rc=${PIPESTATUS[0]}
fi
log "=== STAGE 2 DONE rc=$pre_rc ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 3 — wiki ppl @4096 and @2048
# ==========================================================================
log "=== STAGE 3: wiki ppl ===" | tee -a "$PROG"
for SEQ in 4096 2048; do
  o="$OUT/$VAR/ppl${SEQ}"; mkdir -p "$o"
  CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$HARNESS_PPL" \
      --model "$EXPDIR/$VAR" --output_dir "$o" --wiki_text "$WIKI" \
      --seqlen "$SEQ" --wiki_tokens 100000000 --device cuda:0 2>&1 | tee "$o/ppl${SEQ}.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "PPL@${SEQ} failed -- a row missing a PPL cell is how the
      2048-vs-4096 mixup happened. Not continuing."
done
log "=== STAGE 3 DONE ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 4 — union-9 zero-shot. Invocation byte-identical to the completed cells
#           except `pretrained`. --batch_size auto is DELIBERATE: the existing rows
#           record batch_size="auto"; hard-coding 64 is a different invocation.
# ==========================================================================
log "=== STAGE 4: union-9 zero-shot on GPUs $GPUS ===" | tee -a "$PROG"
o="$OUT/$VAR"; mkdir -p "$o/lm_eval_out"
CUDA_VISIBLE_DEVICES="$GPUS" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
"$LM_EVAL" \
  --model hf \
  --model_args "pretrained=$EXPDIR/$VAR,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
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
# STAGE 5 — aggregate + ★ PER-TASK n_scored == expected ASSERTION
# ==========================================================================
log "=== STAGE 5: aggregate + completeness assertion ===" | tee -a "$PROG"
"$PY" "$AGG9" --lm-eval-out "$o/lm_eval_out" --output "$o/zeroshot_union9.json" \
    --model "slorb_ladder_rung${RUNG}" 2>&1 | tee -a "$PROG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "aggregation failed -- row INVALID, not writing a summary"

CUDA_VISIBLE_DEVICES="" "$PY" - "$RUNG" "$o" "$EXPDIR/$VAR" "$EXP_DENS" "$EXP_LIVE" \
                              "$EXP_PSI" "$QUALIFIES" "$CAPS" "$GPUS" "$pre_rc" \
  <<'PYEOF' 2>&1 | tee -a "$PROG"
import json, sys, pathlib, glob

rung, outdir, expdir, exp_dens, exp_live, exp_psi, qualifies, caps, gpus, pre_rc = sys.argv[1:11]
outdir = pathlib.Path(outdir)

# n recorded by EVERY arm in sparseforge_5b/sparseforge_same_harness_table.json and
# by both token-matched arms. A cell that scores a different number of docs is NOT
# comparable however clean its json looks -- this is the n_scored == expected
# discipline, and it is NOT a NaN check: a silent 6/8-shard merge once produced a
# complete-looking artefact with n_nan == 0 on a CHANGED measurement basis.
EXPECT_N = {"boolq": 3270, "rte": 277, "hellaswag": 10042, "race": 1045, "piqa": 1838,
            "winogrande": 1267, "arc_easy": 2376, "arc_challenge": 1172, "openbookqa": 500}
EXPECT_PRIMARY = {"boolq": "acc", "rte": "acc", "hellaswag": "acc_norm", "race": "acc",
                  "piqa": "acc", "winogrande": "acc", "arc_easy": "acc_norm",
                  "arc_challenge": "acc_norm", "openbookqa": "acc_norm"}
# Anchors already on disk. tau is pre-registered PRE-DATA, 2 x the widest measured
# paired-CI95 half-width (0.8953 pp) among the three completed union-9 contrasts.
R0_UNION9, R1_UNION9 = 62.4335, 57.0678
TAU_PP = 1.79
PASS_BAR = R0_UNION9 - TAU_PP           # 60.6435
CAST_REPRO_AT_50PCT = 62.0919           # the unreachable Pareto reference

res = sorted(glob.glob(str(outdir / "lm_eval_out" / "**" / "results_*.json"),
                       recursive=True))
if not res:
    raise SystemExit("no results_*.json under lm_eval_out -- cannot verify completeness")
doc = json.loads(pathlib.Path(res[-1]).read_text())
results, nsamp = doc.get("results", {}), doc.get("n-samples", {})

problems, per_task = [], {}
for task, want_n in EXPECT_N.items():
    if task not in results:
        problems.append(f"{task}: ABSENT from results")
        continue
    ns = nsamp.get(task) or {}
    got_n = ns.get("effective", ns.get("original"))
    if got_n is None:
        problems.append(f"{task}: n-samples not recorded -- CANNOT assert n_scored")
    elif int(got_n) != want_n:
        problems.append(f"{task}: n_scored={got_n} != expected {want_n} "
                        f"(NOT COMPARABLE to the existing rows)")
    metric = EXPECT_PRIMARY[task]
    key = next((k for k in results[task] if k.split(",")[0] == metric), None)
    if key is None:
        problems.append(f"{task}: primary metric {metric!r} missing "
                        f"(have {sorted(results[task])[:6]})")
        continue
    v = results[task][key]
    if v != v:                                   # NaN
        problems.append(f"{task}: {metric} is NaN")
    per_task[task] = {"n_scored": got_n, "expected_n": want_n,
                      "metric": metric, "value": v}

if problems:
    print("COMPLETENESS ASSERTION FAILED:")
    for p in problems:
        print("  - " + p)
    raise SystemExit("refusing to write a row: per-task n_scored/metric assertion failed. "
                     "A well-formed json on a different measurement basis is the worst "
                     "possible artefact -- it looks citable.")
print(f"[assert] OK all 9 tasks present, n_scored == expected, primary metric non-NaN")

union9 = 100.0 * sum(per_task[t]["value"] for t in EXPECT_N) / 9
mani = json.loads((pathlib.Path(expdir) / "slorb_ladder_manifest.json").read_text())
dens = 100.0 * mani["density_two_matmul_deployment_form"]
if abs(dens - float(exp_dens)) > 5e-4 or mani["live_branch_params"] != int(exp_live):
    raise SystemExit(f"manifest disagrees with the frozen ladder: density {dens:.4f} vs "
                     f"{exp_dens}, branch_live {mani['live_branch_params']} vs {exp_live}")

loss = R0_UNION9 - union9
if qualifies != "yes":
    verdict = (f"NOT A QUALIFYING RUNG (psi {exp_psi} < 0.60)" if qualifies == "no"
               else "CONTROL ARM -- compare to rung P, do not read as a ladder verdict")
elif union9 >= PASS_BAR:
    verdict = f"PASSES the pre-registered bar ({union9:.4f} >= {PASS_BAR:.4f})"
elif loss <= 2 * TAU_PP:
    verdict = (f"INDETERMINATE: loss {loss:.4f} pp is in the (%.2f, %.2f) band = 1-2 CI "
               f"widths at n=1. MUST NOT be called either way." % (TAU_PP, 2 * TAU_PP))
else:
    verdict = f"FAILS: loss {loss:.4f} pp > 2*tau"

row = {
    "b12_rung": rung, "operator": mani["operator"], "coefficients": mani["coefficients"],
    "coarsen_map": mani["coarsen_map"],
    "density_two_matmul_deployment_pct": dens,
    "live_branch_params": mani["live_branch_params"],
    "psi_density_points_given_back": mani["psi_density_points_given_back"],
    "union9_mean_primary_pp": union9,
    "loss_vs_R0_pp": loss,
    "tau_pp": TAU_PP, "pass_bar": PASS_BAR,
    "qualifies_for_rank_claim": qualifies,
    "verdict": verdict,
    "per_task": per_task,
    "n_scored_assertion": "PASSED -- all 9 tasks, per-task n_scored == expected",
    "compute_caps": caps.rstrip(","), "gpus": gpus,
    "verify_2of4_pre_rc": int(pre_rc),
    "is_dense_on_disk": mani.get("is_dense_on_disk"),
    "harness": ("lm_eval 0.4.8, --model hf, dtype=bfloat16, parallelize=True, "
                "add_bos_token=False, --batch_size auto, --num_fewshot 0, --seed 0"),
    "comparable_only_to": "cells measured on compute_cap 10.0 (sm_100/B200)",
    "anchors_on_disk": {"R0_hard_fold_DENSE": R0_UNION9, "R1_hard_drop_2of4": R1_UNION9},
    "must_not_claim": [
        "That this is a training-time rank ablation. It is post-hoc surgery on a "
        "checkpoint trained with SLoRB_k=16 and cannot bound what training with a "
        "larger k would give.",
        "Any placement of this rung in a 2:4 column -- it is FOLDED and dense on disk.",
        f"Pareto superiority over CAST-repro pure 2:4, which scores "
        f"{CAST_REPRO_AT_50PCT} at density EXACTLY 50.0% (below every rung). "
        f"Unreachable by construction.",
        "Any reading of this row without stating n=1 and that tau is 33.4% of the "
        "entire R0-R1 window (5.3657 pp).",
    ],
}
p = outdir / f"slorb_ladder_row_{rung}.json"
p.write_text(json.dumps(row, indent=2))
print(json.dumps({k: v for k, v in row.items() if k != "per_task"}, indent=2))
print(f"[b12] wrote {p}")
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || die "completeness/verdict step failed -- row NOT written"
log "=== STAGE 5 DONE ===" | tee -a "$PROG"

# ==========================================================================
# STAGE 6 — 2:4 verify (POST). Drift across inference is a red flag either way.
# ==========================================================================
log "=== STAGE 6: 2:4 verify $VAR (POST) ===" | tee -a "$PROG"
post_rc=0
if [ -f "$VERIFY" ]; then
  CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$VERIFY" --model "$EXPDIR/$VAR" \
      --sample-layers 12 --seed 0 2>&1 | tee "$OUT/verify_2of4_${VAR}_post.log"
  post_rc=${PIPESTATUS[0]}
fi
[ "$pre_rc" -eq "$post_rc" ] \
  || log "WARN 2:4 gate changed across inference (pre=$pre_rc post=$post_rc) -- investigate" | tee -a "$PROG"
log "=== STAGE 6 DONE rc=$post_rc ===  rung $RUNG COMPLETE" | tee -a "$PROG"
