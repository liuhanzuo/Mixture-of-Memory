#!/usr/bin/env bash
# ============================================================================
# A04 — in-domain held-out PPL at step 150 000 for arm keep7f2, so that the
# PLATEAU rule becomes EVALUABLE at a checkpoint whose capability is already
# measured.
#
# WHY THIS EXISTS
# ---------------
# `A04_STEP100K_PLATEAU_VS_NI_VERDICT.md` §4 / §8 item 1 records a GRID
# ASYMMETRY, and names closing it as GPU work not done there:
#
#   * capability axes are scored at steps {50 000, 100 000, 150 000, 200 000};
#   * in-domain PPL exists at steps {50 000, 100 000, 147 000, 200 000}.
#
# The grids differ at 150 000 (capability yes, PPL no) and at 147 000 (PPL yes,
# capability no). PLATEAU needs PPL, so it is UNDEFINED at step 150 000 and that
# checkpoint cannot form a PLATEAU-vs-NI disagreement even though its NI cells
# already exist. Verbatim: "Closing the 150k side would need an in-domain PPL run
# at step 150 000 (cheap, ~8 shards), which is *not* done here." This driver
# does it.
#
# WHY 150 000 AND NOT 147 000 — THE CHEAPER PAIRING IS IMPOSSIBLE
# ---------------------------------------------------------------
# Pairing at 147 000 would be cheaper (PPL already measured, nothing to run) but
# it CANNOT be done: `step147000.pt` NO LONGER EXISTS ON EITHER DISK.
#   * zwfy6: `find /apdcephfs_zwfy6/share_304376610/pighzliu_code -maxdepth 4
#     -name 'step147000*'` returns NOTHING (2026-08-12). The arm dir holds only
#     step{50000,100000,150000,200000}.pt + final.pt.
#   * wzc1: there is no `*keep7fresh2*` directory at all on the wzc1 disk. And
#     on `.73` the path `/apdcephfs_wzc1` is a SYMLINK to `/apdcephfs_zwfy6`,
#     so a wzc1-looking path there is the same physical file set.
# The 147 000 PPL point is a fossil: it was measured 2026-07-19 (see
# `olmo2_ppl_results/1B_keep7_step147000/` mtimes) while the ckpt still existed,
# and the ckpt was later pruned. Capability at 147 000 is therefore permanently
# unobtainable without retraining, whereas `step150000.pt` IS on disk. So
# step 150 000 is not merely the better target, it is the ONLY target that can
# ever close this side of the bracket.
#
# PROTOCOL — RECOVERED, NOT GUESSED, AND DELIBERATELY IDENTICAL
# -------------------------------------------------------------
# The four archived PPL points were produced by `scripts/_run_olmo2_probe2_ppl_8gpu.sh`
# (git 89d5f15, 2026-07-19). Every runtime parameter below was recovered from
# that launcher AND independently confirmed against the archived summaries'
# own recorded meta (`olmo2_ppl_results/1B_keep7_step{50000,100000,147000,200000}/summary.json`):
#
#     --base_model ../models/OLMo-2-0425-1B     (resolves to
#         /apdcephfs_zwfy6/.../models/OLMo-2-0425-1B == the ckpt's own
#         recorded base_model_path; 16 layers, vocab 100352)
#     --val_path   data/dolmino_now_val.npy     (shape (4096, 2048) uint32)
#     --num_shards 8  --shard_index 0..7        (one process per GPU, windows[g::8])
#     --batch_size 4
#     --limit      NOT PASSED (default 0 = no cap)
#     keep_front_layers=7  n_fresh_layers=2
#     merge via --merge (token-weighted; NEVER a mean of per-shard ppl)
#
# Invariants the archived points all satisfy, asserted below for the new one:
#     n_shards = 8   n_tokens = 8 384 512   n_windows = 4 096
# (4096 windows x 2047 predicted positions = 8 384 512 — the whole val set.)
#
# HARNESS IDENTITY. `scripts/eval_olmo2_probe2_ppl.py` is byte-identical across
# wzc1 and zwfy6 (md5 `12b2dede720410c861eee78fc91e012a`, verified 2026-08-12).
# Three commits touched it after the archived run (89d5f15, 2026-07-19):
#     d380bbc  merge_shards: refuse a silent PARTIAL-shard merge. Guard only —
#              on a complete 8/8 set the arithmetic is unchanged.
#     36ddb1e  adds load_base_model_any_family (NEW function, non-OLMo).
#     7ac9653  adds load_truncated_any_family (NEW function, non-OLMo).
# None of the three modifies `load_pruned_model` or `score_windows` — the two
# functions on this driver's live path. So step 150 000 is scored by the same
# code that scored the archived four. The repo's standing rule is that
# same-arch/same-harness re-runs are BYTE-IDENTICAL, which is exactly why the
# regression config below is a real test and not a formality.
#
# REGRESSION GATE (runs FIRST, deliberately)
# ------------------------------------------
# Config 1 re-measures step 100 000 — an ALREADY-PUBLISHED point
# (16.161295049729876) — under this recovered protocol, into a SEPARATE output
# dir `A04_regress_1B_keep7_step100000` so the archived dir is never touched. If
# it does not reproduce, the step-150 000 number is not comparable to the
# trajectory it is meant to join, and that is the finding. Config 2 is the new
# point. Order is regression-then-new so a failure is visible before the new
# number is trusted.
#
# ARCH: keep/fresh are read FROM THE CKPT META by `load_pruned_model` and the
# CLI values must AGREE or it raises. Passing them explicitly is therefore a
# free assertion that the ckpt really is keep7+fresh2, not a guess.
#
# NODE: zwfy6 only (.73/.82/.104) — that is where the ckpts and the val npy are.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU BS
# usage:
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/active/A04-recovery-certification/code/a04_ppl_step150k_driver.sh \
#     > logs/a04_ppl_step150k.out 2>&1 &
# ============================================================================
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NGPU="${NGPU:-8}"
BS="${BS:-4}"                       # recovered: _run_olmo2_probe2_ppl_8gpu.sh:9
VAL="data/dolmino_now_val.npy"      # recovered: launcher:8 + all four summaries
BASE="../models/OLMo-2-0425-1B"     # recovered: launcher:17-19 + summaries
CKDIR="outputs/olmo2_probe2_1B_keep7fresh2_16card"

# invariants every archived point satisfies
EXP_TOKENS=8384512
EXP_WINDOWS=4096

cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd $PROJECT_ROOT"; exit 9; }
mkdir -p logs olmo2_ppl_results
PROG=logs/a04_ppl_step150k_progress.log
note() { echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

note "START a04_ppl_step150k_driver | root=$PROJECT_ROOT py=$PYTHON_BIN ngpu=$NGPU bs=$BS val=$VAL"
note "harness md5: $(md5sum scripts/eval_olmo2_probe2_ppl.py | awk '{print $1}')"

# ---- refuse to run if the arm's ckpts are not what we think they are -------
for st in 100000 150000; do
  [ -f "$CKDIR/step${st}.pt" ] || { note "FATAL: missing $CKDIR/step${st}.pt"; exit 3; }
done
[ -f "$VAL" ] || { note "FATAL: missing $VAL"; exit 3; }
# 147000 must be ABSENT — if it ever reappears, the cheaper pairing becomes
# possible and the rationale in this header needs revisiting rather than being
# silently outrun by the filesystem.
if [ -f "$CKDIR/step147000.pt" ]; then
  note "NOTE: step147000.pt EXISTS after all -- the 147000 pairing is now"
  note "      possible and is cheaper. Re-read this driver's header before"
  note "      treating 150000 as the only option."
fi

# "output_name|ckpt|role"
CONFIGS=(
  "A04_regress_1B_keep7_step100000|$CKDIR/step100000.pt|REGRESSION (must reproduce 16.161295049729876)"
  "1B_keep7_step150000|$CKDIR/step150000.pt|NEW POINT (closes the 150k side of the bracket)"
)

T_ALL0=$(date +%s)
for row in "${CONFIGS[@]}"; do
  NAME="${row%%|*}"; rest="${row#*|}"
  CKPT="${rest%%|*}"; ROLE="${rest#*|}"
  note "=========================================================="
  note "CONFIG $NAME | ckpt=$CKPT | $ROLE"
  T0=$(date +%s)
  for g in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$g "$PYTHON_BIN" scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --keep_front_layers 7 --n_fresh_layers 2 \
      --val_path "$VAL" --num_shards "$NGPU" --shard_index "$g" \
      --batch_size "$BS" --output_name "$NAME" \
      > "logs/a04_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  T1=$(date +%s)
  note "$NAME shards done in $((T1 - T0)) s; merging"
  "$PYTHON_BIN" scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1 | tee -a "$PROG"

  # ---- assert the measurement basis, per the repo's partial-merge lesson ----
  "$PYTHON_BIN" - "$NAME" "$EXP_TOKENS" "$EXP_WINDOWS" <<'PY' 2>&1 | tee -a "$PROG"
import json, sys
name, exp_tok, exp_win = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = json.load(open(f"olmo2_ppl_results/{name}/summary.json"))
bad = []
if d["n_shards"] != 8:      bad.append(f"n_shards={d['n_shards']} != 8")
if d["n_tokens"] != exp_tok: bad.append(f"n_tokens={d['n_tokens']} != {exp_tok}")
if d["n_windows"] != exp_win: bad.append(f"n_windows={d['n_windows']} != {exp_win}")
m = d["meta"]
if m.get("keep_front_layers") != 7:  bad.append(f"keep={m.get('keep_front_layers')} != 7")
if m.get("n_fresh_layers") != 2:     bad.append(f"fresh={m.get('n_fresh_layers')} != 2")
if m.get("num_hidden_layers") != 9:  bad.append(f"layers={m.get('num_hidden_layers')} != 9")
if m.get("val_path") != "data/dolmino_now_val.npy": bad.append(f"val={m.get('val_path')}")
if bad:
    print(f"[ASSERT-FAIL] {name}: " + "; ".join(bad))
    sys.exit(4)
print(f"[ASSERT-OK] {name}: ppl={d['ppl']!r} avg_nll={d['avg_nll']!r} "
      f"n_shards=8 n_tokens={d['n_tokens']} n_windows={d['n_windows']} "
      f"ckpt_step={m.get('ckpt_step')}")
PY
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { note "FATAL: basis assertion failed for $NAME (rc=$rc)"; exit "$rc"; }
done
T_ALL1=$(date +%s)

# ---- the regression verdict, stated numerically ---------------------------
"$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$PROG"
import json
PUB = 16.161295049729876          # archived 1B_keep7_step100000
new = json.load(open("olmo2_ppl_results/A04_regress_1B_keep7_step100000/summary.json"))
old = json.load(open("olmo2_ppl_results/1B_keep7_step100000/summary.json"))
d150 = json.load(open("olmo2_ppl_results/1B_keep7_step150000/summary.json"))
print("=" * 74)
print("REGRESSION: step100000 re-measured under the recovered protocol")
print(f"  published (evidence json)   : {PUB!r}")
print(f"  archived summary.json       : {old['ppl']!r}")
print(f"  re-measured now             : {new['ppl']!r}")
print(f"  |re-measured - published|   : {abs(new['ppl'] - PUB):.3e}")
print(f"  sum_nll  archived / now     : {old['sum_nll']!r} / {new['sum_nll']!r}")
print(f"  BYTE-IDENTICAL sum_nll?     : {old['sum_nll'] == new['sum_nll']}")
print(f"  REPRODUCES (<1e-9)          : {abs(new['ppl'] - PUB) < 1e-9}")
print("-" * 74)
print(f"NEW POINT step150000 ppl      : {d150['ppl']!r}")
print(f"  avg_nll                     : {d150['avg_nll']!r}")
print(f"  sum_nll                     : {d150['sum_nll']!r}")
print("=" * 74)
PY

note "ALL DONE in $((T_ALL1 - T_ALL0)) s wall (both configs, ${NGPU} GPUs)"
note "DONE_MARKER a04_ppl_step150k"
