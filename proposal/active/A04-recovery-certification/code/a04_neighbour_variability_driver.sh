#!/usr/bin/env bash
# ============================================================================
# A04 — NEIGHBOUR VARIABILITY: 4-axis capability scoring of TIGHT checkpoint
# clusters, so the "hand-picked checkpoint" loophole can be PRICED.
#
# WHY THIS EXISTS
# ---------------
# `A04_KEEP14_TRAJECTORY_NI_VERDICT.md` 7.2 (commit 517c8d2) established, on
# ONE arm at 25 500-step spacing, that
#
#     "a certification rule evaluated at a single arbitrary checkpoint can
#      return a BETTER verdict than a later checkpoint of the same run, so any
#      future accept obtained at a hand-picked checkpoint must be shown to
#      survive its neighbours."
#
# That rule-level claim has two untested legs, and this driver supplies the
# checkpoints for both:
#
#   LEG A (the quantitative gap the claim never measured).  Hand-picking does
#   not happen at 25 500-step spacing -- it happens between ADJACENT saves.  The
#   checkpoint-to-checkpoint variability of the margin has never been measured,
#   so the requirement "must survive its neighbours" carries NO tolerance.  Two
#   500-step-spaced clusters of keep8+fresh2 give the range directly.
#
#   LEG B (cross-arm replication).  The popqa 128k->153.5k regression is n=1
#   arm.  shortgpt16 has ckpts at the SAME three steps, so it is the one
#   available independent-arm replication.
#
# ARM/AXIS FACTS ESTABLISHED BEFORE SPENDING GPU (2026-08-13, both disks)
# ----------------------------------------------------------------------
#  * keep8fresh2 is keep_front=8 n_fresh=2 num_hidden_layers=10, 113 tensors
#    (arch_meta.json + every ckpt meta).  It is a DIFFERENT ARCHITECTURE from
#    keep14fresh2 (14+2 = 16 layers, 179 tensors).  So Leg A measures
#    neighbour variability *within an arm*, which is what the claim is about,
#    but its absolute margins are NOT comparable to the keep14 curve and must
#    never be tabulated alongside it as if they were another rung of it.
#  * All six Leg A ckpts load, are fp32, 113 tensors, and are DISTINCT WEIGHTS
#    (per-tensor sha256 of lm_head/embed_tokens/layers.0.q_proj all differ; the
#    f64 sum of every parameter differs).  Byte size is NOT identity here:
#    124000 and 124500 share a size (34152197522) as do 125000/130000/130500/
#    131000 (34152196306).
#  * ! CLUSTER 1 STRADDLES A RESUME BOUNDARY.  step124000 and step124500 were
#    written by the .73 run of 2026-08-08 (`logs/olmo2_7B_keep8fresh2_resume200k_73.log`,
#    which resumed from step121000_full.pt and died at 20:26 with a TCPStore
#    error after saving 124500).  step125000 was written by a DIFFERENT process
#    on .82 four days later (`logs/keep8_resume_82_launch.out`, 2026-08-12
#    00:34, resumed FROM step124500.pt).  The trainer restores optimizer state
#    and RNG but re-creates the loader (`data_iter = iter(loader)` after
#    `sampler.set_epoch(epoch)`) -- it does NOT fast-forward within the epoch --
#    so the 124500->125000 interval saw a DIFFERENT DATA ORDER than an
#    uninterrupted 500 steps would have.  CLUSTER 2 (130000/130500/131000) is
#    entirely inside the single .82 process and has NO such boundary.  The two
#    clusters are therefore reported SEPARATELY and cluster 2 is the clean
#    measurement; comparing them is itself informative (does a resume seam widen
#    the neighbour range?).  Recording this is the whole point of pre-checking.
#  * ! shortgpt16/step128000.pt on zwfy6 is CORRUPT: 7 755 268 096 B (vs
#    48 724 473 978 B for its 153500/200000 siblings), and both `zipfile` and
#    `torch.load` fail with "failed finding central directory ... checkpoint is
#    corrupted".  The wzc1 copy of the same file IS intact (48 724 473 978 B,
#    731 zip entries, testzip clean).  Leg B therefore needs that file staged
#    across disks first (measured 16 MiB/s => ~48 min); this driver takes
#    SG16_CKPT_128000 as an override so the staged path can be passed in, and
#    REFUSES to substitute the corrupt file.
#  * shortgpt16 is keep_front=16 n_fresh=0, keep_layer_indices
#    [0..12,16,17,31] -- a NON-CONTIGUOUS layer selection.  Its step128000/
#    153500/200000 are the same integers as keep14's, but "step" is only
#    synonymous as an optimizer-step count: shortgpt16 reaches epoch 2 at
#    153500 and epoch 3 at 200000, keep14 reaches epoch 2 at 153500 and its
#    endpoint ckpt records no epoch.  Same step != same data seen, and the two
#    arms differ in damage geometry as well as depth.  Leg B is a REPLICATION
#    OF THE PHENOMENON (does popqa dip over these steps on another arm), never
#    a matched pairwise comparison.
#
# PROTOCOL — FROZEN, IDENTICAL TO THE keep14 TRAJECTORY SCAN
# ---------------------------------------------------------
# cb bs=32, mmlu bs=16, add_bos=0, num_shards=8, max_new_tokens=32, greedy,
# base LM, chat_template unset (neither harness HAS a chat-template code path).
# Same anchor (vanilla ../models/OLMo-2-1124-7B), same harnesses
# (scripts/eval_olmo2_closedbook_qa.py md5 2ed41993…, eval_olmo2_mmlu_content.py
# md5 fe4a62db…).  Batch size is NOT free:
# `full32_rescore_v2_20260812.sensitivity_bs48_probe` measured bs32->bs48
# flipping 12/14267 popqa and 10/3610 nq_open items.  This driver echoes
# `DRIVER START ... mmlu_bs=<N> cb_bs=<N>` and a per-axis `START ... bs=<N>` so
# the analysis can parse the ACTUAL invocation (summary.json:meta records
# neither batch_size nor chat_template -- A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md).
#
# ARCH IS ASSERTED, NEVER GUESSED: keep/fresh are read from the ckpt meta by
# `load_pruned_model` and a CLI mismatch raises, so passing KEEP_FRONT/N_FRESH
# explicitly is a free assertion.  A PARTIAL SHARD SET IS A FAILURE, never a
# silent skip (the repo has been corrupted before by a merged 5-of-8 set).
#
# NODES: zwfy6 only (.73 / .82).  .104 is running paperC Qwen3 heal and
# LOCAL/.21 are running SparseForge #246 -- all outside this dispatch's budget.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU MMLU_BS CB_BS ARM STEPS TAG_PREFIX
#      KEEP_FRONT N_FRESH SG16_CKPT_128000
# usage (Leg A cluster 2, on .73):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   ARM=olmo2_probe2_7B_keep8fresh2 KEEP_FRONT=8 N_FRESH=2 \
#   TAG_PREFIX=A04_7B_keep8f2 STEPS="130000 130500 131000" \
#   setsid nohup bash proposal/active/A04-recovery-certification/code/a04_neighbour_variability_driver.sh \
#     > logs/a04_nbr_keep8_c2.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-1124-7B}"
NGPU="${NGPU:-8}"
MMLU_BS="${MMLU_BS:-16}"      # frozen, matches the anchor + the keep14 scan
CB_BS="${CB_BS:-32}"          # frozen, matches the anchor + the keep14 scan
ARM="${ARM:?set ARM=olmo2_probe2_7B_keep8fresh2 | olmo2_probe2_7B_shortgpt16}"
STEPS="${STEPS:?set STEPS='130000 130500 131000'}"
TAG_PREFIX="${TAG_PREFIX:?set TAG_PREFIX=A04_7B_keep8f2 | A04_7B_sg16}"
KEEP_FRONT="${KEEP_FRONT:?set KEEP_FRONT}"
N_FRESH="${N_FRESH:?set N_FRESH}"
# Leg B only: zwfy6's shortgpt16/step128000.pt is CORRUPT (see header). Pass the
# path of the wzc1 copy staged onto this disk. Never silently fall back.
SG16_CKPT_128000="${SG16_CKPT_128000:-}"
CKDIR="outputs/$ARM"
PROG="logs/a04_nbr_progress_$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d ' ').log"

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs olmo2_mmlu_content_results olmo2_closedbook_results

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- refuse to share the node -----------------------------------------------
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
  # Leg B override for the one file that is corrupt on this disk.
  if [ -n "$SG16_CKPT_128000" ] && [ "$s" = "128000" ]; then
    echo "$SG16_CKPT_128000"; return
  fi
  if   [ -f "$CKDIR/step${s}.pt" ];       then echo "$CKDIR/step${s}.pt"
  elif [ -f "$CKDIR/keep14_step${s}.pt" ]; then echo "$CKDIR/keep14_step${s}.pt"
  else echo ""; fi
}

run_mmlu() {  # $1=output_name $2=ckpt
  local NAME="$1" CK="$2" RD="olmo2_mmlu_content_results/$1"
  if [ -f "$RD/summary.json" ]; then note "mmlu SKIP $NAME (summary.json exists)"; return 0; fi
  note "mmlu START $NAME bs=$MMLU_BS"
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data --content_desc full \
    > "logs/a04nbr_mmlu_${NAME}_prepare.log" 2>&1 || true
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --content_desc full --num_shards $NGPU --shard_index $g \
      --batch_size $MMLU_BS --add_bos 0 \
      --output_name "$NAME" > "logs/a04nbr_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
  note "mmlu shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "mmlu ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" --n_boot 10000 \
      >> "logs/a04nbr_mmlu_${NAME}_merge.log" 2>&1
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
      --output_name "$NAME" > "logs/a04nbr_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "closedbook shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "closedbook ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04nbr_cb_${NAME}_merge.log" 2>&1
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
      --output_name "$NAME" > "logs/a04nbr_nq_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "nq_open shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "nq_open ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04nbr_nq_${NAME}_merge.log" 2>&1
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
  # torch.load probe: byte SIZE IS NOT IDENTITY on this arm (124000/124500 share
  # a size; 125000/130000/130500/131000 share another), and one sibling file in
  # this repo is a truncated zip that `ls -l` cannot distinguish from healthy.
  # So the meta step / arch are asserted from the loaded object before 8 GPUs
  # are spent, and the zip central directory is exercised by the load itself.
  probe=$($PY - "$CK" "$S" "$KEEP_FRONT" "$N_FRESH" <<'EOF' 2>/dev/null || echo bad
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
assert int(ck["step"]) == int(sys.argv[2]), f"meta step {ck['step']} != requested {sys.argv[2]}"
assert int(ck["keep_front_layers"]) == int(sys.argv[3]), ck["keep_front_layers"]
assert int(ck["n_fresh_layers"]) == int(sys.argv[4]), ck["n_fresh_layers"]
assert int(ck["num_hidden_layers"]) == int(sys.argv[3]) + int(sys.argv[4])
n = len(ck["model_state"])
assert n in (113, 179), n
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
