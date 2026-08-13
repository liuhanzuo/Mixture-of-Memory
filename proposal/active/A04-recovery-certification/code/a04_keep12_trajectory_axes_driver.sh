#!/usr/bin/env bash
# ============================================================================
# A04 — keep12+fresh2 DENSE TRAJECTORY: 4-axis capability scoring of an
# 8-point, exactly-5000-step-spaced grid, so MONOTONICITY of the NI margin can
# be tested as a TREND rather than as a sign.
#
# WHY THIS EXISTS
# ---------------
# `A04_KEEP12_TRAJECTORY_PREREG.md` states claim P: the NI margin of a damaged,
# healing arm wanders non-monotonically along training with an amplitude
# comparable to Delta, so a single-point accept is uninterpretable without its
# neighbourhood.  P's three existing legs are each defective:
#   * keep14 trajectory   -- 3 points, 25 500-step UNEVEN spacing.
#   * neighbour variability -- 7 of 8 decision-axis ranges fall INSIDE the
#     noise gate E[range of 3] = 1.6926*sigma; only 1 of 8 clears it.
#   * full32 trajectory   -- ZERO-DAMAGE CPT arm, not a recovery arm at all.
# What P has never had is a genuinely damaged recovery arm on a DENSE, EVENLY
# SPACED grid with enough points to fit a trend.  keep12+fresh2 supplies it.
#
# ARM FACTS ESTABLISHED BEFORE SPENDING GPU (2026-08-13, zwfy6)
# ------------------------------------------------------------
#  * keep12fresh2 is keep_front=12 n_fresh=2 num_hidden_layers=14, 157 tensors
#    (arch_meta.json + every ckpt meta).  It is a DIFFERENT DAMAGE LEVEL from
#    keep14fresh2 (14+2 = 16 layers, 179 tensors) and from keep8fresh2 (8+2 =
#    10 layers, 113 tensors).  So this is a CROSS-ARM test of P, not extra
#    points on the arm that generated P -- and absolute margins from the three
#    arms are NOT rungs of one ladder and must never be tabulated as such.
#  * ALL ELEVEN ckpts load, are fp32, 157 tensors, epoch=1, and are DISTINCT
#    WEIGHTS: per-tensor sha256 of lm_head / embed_tokens / layers.0.q_proj all
#    differ pairwise, AND the float64 sum of every parameter differs at every
#    step (68640.843 / 67443.382 / 66041.978 / 65032.452 / 63042.541 /
#    62267.044 / 61244.476 / 60618.238 / 59721.344 / 59683.796 / 59540.192 for
#    124000/130000/.../166000).  BYTE SIZE IS NOT IDENTITY on this arm:
#    130000..166000 all share 43 867 049 986 B and only step124000 differs
#    (43 867 047 810 B).  A repo sibling (shortgpt16/step128000.pt on zwfy6) is
#    a TRUNCATED zip that `ls -l` cannot distinguish from healthy, so the meta
#    step / arch are asserted from the LOADED object before 8 GPUs are spent.
#  * ! NO RESUME SEAM ANYWHERE IN 124000 -> 166000.  This was checked FIRST,
#    because `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` 1.2 found that a 500-step
#    neighbourhood CAN straddle a process boundary (the trainer restores
#    optimizer+RNG but rebuilds the loader via
#    `sampler.set_epoch(epoch); data_iter = iter(loader)` WITHOUT
#    fast-forwarding inside the epoch, so the crossing interval sees a different
#    data order).  `logs/olmo2_7B_keep12fresh2_resume200k_v2.log` has exactly
#    ONE process start (`[seed] set_seed(42)` 2026-08-08 13:58:02, one
#    `[resume] loading ckpt ... step124000.pt`, one
#    `[resume] sampler.set_epoch(1)`) and every one of the eleven ckpts is
#    saved by it (grid saves at log lines 348/626/904/1182/1460/1738/2016/2294,
#    plus 2323/2351 for 165500/166000; the log's last line is step 166020).
#    So the whole grid is ONE uninterrupted data order and the 500-step triple
#    {165000,165500,166000} is clean BY CONSTRUCTION -- unlike keep8 cluster 1.
#
# PROTOCOL — FROZEN, IDENTICAL TO THE keep14 AND NEIGHBOUR SCANS
# --------------------------------------------------------------
# cb bs=32, mmlu bs=16, add_bos=0, num_shards=8, max_new_tokens=32, greedy,
# base LM, chat_template unset (neither harness HAS a chat-template code path).
# Same anchor (vanilla ../models/OLMo-2-1124-7B), same harnesses
# (scripts/eval_olmo2_closedbook_qa.py md5 2ed41993241226c795a3ca38375933f7,
# eval_olmo2_mmlu_content.py md5 fe4a62dbdf884a1e2aedc6ed26887b4e -- verified
# IDENTICAL on .73 and .82).  Batch size is NOT free:
# `full32_rescore_v2_20260812.sensitivity_bs48_probe` measured bs32->bs48
# flipping 12/14267 popqa and 10/3610 nq_open items.  This driver echoes
# `DRIVER START ... mmlu_bs=<N> cb_bs=<N>` and a per-axis `START ... bs=<N>` so
# the analysis can parse the ACTUAL invocation -- summary.json:meta records
# NEITHER batch_size NOR chat_template (A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md),
# and the analysis re-verifies these from THIS dispatch's logs rather than
# inheriting keep14's.
#
# ARCH IS ASSERTED, NEVER GUESSED: keep/fresh are read from the ckpt meta by
# `load_pruned_model` and a CLI mismatch raises, so passing KEEP_FRONT/N_FRESH
# explicitly is a free assertion.  A PARTIAL SHARD SET IS A FAILURE, never a
# silent skip (this repo has been corrupted by a merged 5-of-8 set).
#
# NODES: zwfy6 only (.73 / .82).  .104 is running paperC Qwen3 heal and
# LOCAL/.21 are running SparseForge #246 -- all OUTSIDE this dispatch's budget.
# The driver REFUSES to start if >8000 MiB of GPU memory is already held.
# NOTE: scoring may be split across .73/.82 (it is deterministic given the
# harness), but ALL BOOTSTRAP STATISTICS must be computed on ONE node -- numpy's
# multinomial differs between 2.5.1 (.73) and 2.4.6 (.82) in 19/10000 rows
# (A04_NEIGHBOUR_VARIABILITY_VERDICT.md 4.1).  This verdict computes on .73.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU MMLU_BS CB_BS ARM STEPS TAG_PREFIX
#      KEEP_FRONT N_FRESH
# usage (grid first half, on .73):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   ARM=olmo2_probe2_7B_keep12fresh2 KEEP_FRONT=12 N_FRESH=2 \
#   TAG_PREFIX=A04_7B_keep12f2 STEPS="130000 135000 140000 145000 150000 155000" \
#   setsid nohup bash proposal/active/A04-recovery-certification/code/a04_keep12_trajectory_axes_driver.sh \
#     > logs/a04_keep12_traj_73.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-1124-7B}"
NGPU="${NGPU:-8}"
MMLU_BS="${MMLU_BS:-16}"      # frozen, matches the anchor + keep14 + neighbour scans
CB_BS="${CB_BS:-32}"          # frozen, matches the anchor + keep14 + neighbour scans
ARM="${ARM:?set ARM=olmo2_probe2_7B_keep12fresh2}"
STEPS="${STEPS:?set STEPS='130000 135000 ...'}"
TAG_PREFIX="${TAG_PREFIX:?set TAG_PREFIX=A04_7B_keep12f2}"
KEEP_FRONT="${KEEP_FRONT:?set KEEP_FRONT}"
N_FRESH="${N_FRESH:?set N_FRESH}"
CKDIR="outputs/$ARM"
PROG="logs/a04_keep12_progress_$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d ' ').log"

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs olmo2_mmlu_content_results olmo2_closedbook_results

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- refuse to share the node (another agent's run must not be corrupted, and
# --- ours must not be OOM'd by one) -----------------------------------------
gpu_free_or_die() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [ "$used" -gt 8000 ]; then
    note "REFUSE: ${used}MiB of GPU memory held by another process -- not sharing the node"
    exit 8
  fi
}
gpu_free_or_die

resolve_ckpt() {
  local s="$1"
  if [ -f "$CKDIR/step${s}.pt" ]; then echo "$CKDIR/step${s}.pt"; else echo ""; fi
}

run_mmlu() {  # $1=output_name $2=ckpt $3=step
  local NAME="$1" CK="$2" RD="olmo2_mmlu_content_results/$1"
  if [ -f "$RD/summary.json" ]; then note "mmlu SKIP $NAME (summary.json exists)"; return 0; fi
  note "mmlu START $NAME bs=$MMLU_BS"
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data --content_desc full \
    > "logs/a04k12_mmlu_${NAME}_prepare.log" 2>&1 || true
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --content_desc full --num_shards $NGPU --shard_index $g \
      --batch_size $MMLU_BS --add_bos 0 \
      --output_name "$NAME" > "logs/a04k12_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
  note "mmlu shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "mmlu ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" --n_boot 10000 \
      >> "logs/a04k12_mmlu_${NAME}_merge.log" 2>&1
  EXP_KEEP=$KEEP_FRONT EXP_FRESH=$N_FRESH EXP_STEP="$3" \
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,os,sys
s=json.load(open(sys.argv[1])); exp=14042
assert s["n"]==exp, f"MMLU n={s['n']} != expected {exp}"
assert s["n_valid"]+s["n_nan"]==exp, "n_valid+n_nan != n"
assert s["n_nan"]==0, f"n_nan={s['n_nan']} -- paired analysis needs an identical valid item set"
assert s["n_shards"]==8, f"n_shards={s['n_shards']}"
m=s["meta"]
assert m["mode"]=="pruned", m
kf,nf=int(os.environ["EXP_KEEP"]),int(os.environ["EXP_FRESH"])
assert m["keep_front_layers"]==kf and m["n_fresh_layers"]==nf, m
assert m["num_hidden_layers"]==kf+nf, m
assert int(m["ckpt_step"])==int(os.environ["EXP_STEP"]), \
    f"ckpt_step={m['ckpt_step']} != requested {os.environ['EXP_STEP']}"
# add_bos must be FALSE. `is False`, NOT `is not True`: the latter passes
# silently on None, which is the project-wide add_bos/chat_template bug.
assert m["add_bos"] is False, f"add_bos={m['add_bos']!r} -- base protocol requires False"
assert m["content_desc"]=="full", m
print(f"OK n={s['n']} valid={s['n_valid']} nan={s['n_nan']} step={m['ckpt_step']} "
      f"letter={s['letter_acc']:.6f} content_norm={s['content_norm_acc']:.6f}")
EOF
  note "mmlu DONE $NAME"
}

run_cb() {  # $1=output_name $2=ckpt $3=step  (popqa+triviaqa in ONE dir)
  local NAME="$1" CK="$2" RD="olmo2_closedbook_results/$1"
  if [ -f "$RD/summary.json" ]; then note "closedbook SKIP $NAME (summary.json exists)"; return 0; fi
  note "closedbook START $NAME bs=$CB_BS"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --tasks popqa,triviaqa --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a04k12_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "closedbook shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "closedbook ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04k12_cb_${NAME}_merge.log" 2>&1
  EXP_KEEP=$KEEP_FRONT EXP_FRESH=$N_FRESH EXP_STEP="$3" \
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,os,sys
s=json.load(open(sys.argv[1]))
exp={"popqa":14267,"triviaqa":17944}
assert s["n_shards"]==8, s["n_shards"]
m=s["meta"]; kf,nf=int(os.environ["EXP_KEEP"]),int(os.environ["EXP_FRESH"])
assert m["mode"]=="pruned" and m["keep_front_layers"]==kf and m["n_fresh_layers"]==nf, m
assert m["num_hidden_layers"]==kf+nf, m
assert int(m["ckpt_step"])==int(os.environ["EXP_STEP"]), \
    f"ckpt_step={m['ckpt_step']} != requested {os.environ['EXP_STEP']}"
assert m["add_bos"] is False, f"add_bos={m['add_bos']!r} -- base protocol requires False"
assert m["max_new_tokens"]==32, m
for t,e in exp.items():
    v=s["tasks"][t]
    assert not v.get("skipped"), f"{t} skipped: {v.get('error')}"
    assert v["n"]==e, f"{t} n={v['n']} != expected {e}"
    print(f"OK {t} n={v['n']} em={v['em']:.8f} contains={v['contains']:.6f} maj_em={v['majority_em']:.6f}")
EOF
  note "closedbook DONE $NAME"
}

run_nq() {  # $1=base_output_name $2=ckpt $3=step -- SEPARATE `_nqopen` dir
  local NAME="${1}_nqopen" CK="$2"; local RD="olmo2_closedbook_results/$NAME"
  if [ -f "$RD/summary.json" ]; then note "nq_open SKIP $NAME (summary.json exists)"; return 0; fi
  note "nq_open START $NAME bs=$CB_BS"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --tasks nq_open --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a04k12_nq_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "nq_open shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "nq_open ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04k12_nq_${NAME}_merge.log" 2>&1
  EXP_KEEP=$KEEP_FRONT EXP_FRESH=$N_FRESH EXP_STEP="$3" \
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,os,sys
s=json.load(open(sys.argv[1]))
assert s["n_shards"]==8, s["n_shards"]
m=s["meta"]; kf,nf=int(os.environ["EXP_KEEP"]),int(os.environ["EXP_FRESH"])
assert m["mode"]=="pruned" and m["keep_front_layers"]==kf and m["n_fresh_layers"]==nf, m
assert int(m["ckpt_step"])==int(os.environ["EXP_STEP"]), \
    f"ckpt_step={m['ckpt_step']} != requested {os.environ['EXP_STEP']}"
assert m["add_bos"] is False, f"add_bos={m['add_bos']!r} -- base protocol requires False"
assert m["max_new_tokens"]==32, m
v=s["tasks"]["nq_open"]
assert not v.get("skipped"), f"nq_open skipped: {v.get('error')}"
assert v["n"]==3610, f"nq_open n={v['n']} != expected 3610"
print(f"OK nq_open n={v['n']} em={v['em']:.8f} contains={v['contains']:.6f} maj_em={v['majority_em']:.6f}")
EOF
  note "nq_open DONE $NAME"
}

note "DRIVER START on $(hostname) ngpu=$NGPU mmlu_bs=$MMLU_BS cb_bs=$CB_BS arm=$ARM steps='$STEPS' keep_front=$KEEP_FRONT n_fresh=$N_FRESH"
rc=0
for S in $STEPS; do
  CK="$(resolve_ckpt "$S")"
  if [ -z "$CK" ]; then note "[skip] step${S}: no ckpt in $CKDIR"; rc=1; continue; fi
  # torch.load probe: byte SIZE IS NOT IDENTITY on this arm (130000..166000 all
  # share 43867049986 B), and a sibling file in this repo is a truncated zip
  # that `ls -l` cannot distinguish from healthy. So the meta step / arch are
  # asserted from the LOADED object before 8 GPUs are spent, and the zip central
  # directory is exercised by the load itself.
  probe=$($PY - "$CK" "$S" "$KEEP_FRONT" "$N_FRESH" <<'EOF' 2>/dev/null || echo bad
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
assert int(ck["step"]) == int(sys.argv[2]), f"meta step {ck['step']} != requested {sys.argv[2]}"
assert int(ck["keep_front_layers"]) == int(sys.argv[3]), ck["keep_front_layers"]
assert int(ck["n_fresh_layers"]) == int(sys.argv[4]), ck["n_fresh_layers"]
assert int(ck["num_hidden_layers"]) == int(sys.argv[3]) + int(sys.argv[4])
n = len(ck["model_state"])
assert n == 157, n
print("ok")
EOF
)
  if [ "$probe" != "ok" ]; then note "[skip] step${S}: ckpt probe FAILED for $CK (corrupt / wrong step / wrong arch)"; rc=1; continue; fi
  note "ckpt OK step=$S -> $CK"
  gpu_free_or_die
  TAG="${TAG_PREFIX}_step${S}"
  t0=$(date +%s)
  run_cb   "$TAG" "$CK" "$S" || rc=1
  run_nq   "$TAG" "$CK" "$S" || rc=1
  run_mmlu "$TAG" "$CK" "$S" || rc=1
  note "step${S} ALL 4 AXES DONE in $(( $(date +%s) - t0 ))s"
done
note "DRIVER END rc=$rc"
exit $rc
