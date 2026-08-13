#!/usr/bin/env bash
# ============================================================================
# A04 -- KEEP10 NEIGHBOUR RANGE: 4-axis capability scoring of a SECOND arm's
# 500-step checkpoint triple, so the neighbour-range loophole priced on ONE arm
# (keep8+fresh2) can be tested for arm-independence.
#
# WHY THIS EXISTS
# ---------------
# `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` measured the NI-margin range across
# three 500-step-spaced checkpoints of keep8+fresh2 and got SIX ranges, of which
# EXACTLY ONE crossed the item-noise gate (triviaqa, 1.1202 pp = 1.70x the range
# pure item noise would produce). `A04_GATE_DESIGN.md` 2.0.2's neighbour
# precondition and 2.5's per-axis tolerance both rest on that ONE cell, and 2.5
# says so explicitly: "These are one-arm numbers and should be widened if a
# second arm is ever measured."
#
# This is that second arm. Pre-registration (committed BEFORE this ran):
# `A04_KEEP10_NEIGHBOUR_RANGE_PREREG.md`.
#
# ARM/CKPT FACTS ESTABLISHED BEFORE SPENDING GPU (2026-08-13, zwfy6)
# ------------------------------------------------------------------
#  * keep10fresh2 = keep_front=10 n_fresh=2 num_hidden_layers=12, from
#    `outputs/olmo2_probe2_7B_keep10fresh2/arch_meta.json` (n_params
#    3,250,786,304; base `/apdcephfs_zwfy6/.../models/OLMo-2-1124-7B`). It is a
#    THIRD architecture: keep8fresh2 is 10 layers, keep12fresh2 is 14,
#    keep14fresh2 is 16. Absolute margins are NOT rungs of one ladder and are
#    never tabulated as such -- only the within-arm RANGES are compared.
#  * step89000 / step89500 / step90000 are 39,009,621,855 B each. BYTE SIZE IS
#    NOT IDENTITY (keep8's 130000/130500/131000 also share one size), so the
#    per-step probe below asserts the LOADED meta's step/keep/fresh/depth, and
#    the torch.load itself exercises the zip central directory -- the exact
#    failure mode that made shortgpt16/step128000.pt (7.7 GB truncated write)
#    indistinguishable from healthy under `ls -l`.
#  * NO RESUME SEAM. `logs/olmo2_7B_keep10fresh2_resume200k_73.log` has exactly
#    ONE `[resume] loading ckpt ... step86500.pt` banner (03:57:09, epoch=0,
#    max_steps=200000), then `saved ... step89000.pt` at 08:44:52 (line 154),
#    `step89500.pt` 09:42:19 (line 182), `step90000.pt` 10:39:43 (line 210).
#    The process died at 11:15 on a TCPStore error -- AFTER all three saves. So
#    one process, one loader, continuous data order: this triple is the CLEAN
#    neighbourhood that keep8's cluster 1 was not.
#
# PROTOCOL -- FROZEN, IDENTICAL TO keep8 LEG A / keep12 / keep14
# -------------------------------------------------------------
# cb bs=32, mmlu bs=16, add_bos=0, num_shards=8, max_new_tokens=32, greedy, base
# LM, chat_template unset (neither harness HAS a chat-template code path).
# Harness md5 verified identical on this disk: eval_olmo2_closedbook_qa.py
# 2ed41993241226c795a3ca38375933f7, eval_olmo2_mmlu_content.py
# fe4a62dbdf884a1e2aedc6ed26887b4e. Batch size is NOT free
# (`full32_rescore_v2_20260812.sensitivity_bs48_probe`: bs32->bs48 flipped
# 12/14267 popqa and 10/3610 nq_open), and `summary.json:meta` records NEITHER
# batch_size NOR chat_template (A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md) -- so this
# driver echoes `DRIVER START ... mmlu_bs=<N> cb_bs=<N>` and a per-axis
# `START ... bs=<N>` and the analysis PARSES THIS LOG as the protocol evidence.
#
# `add_bos` is asserted `is False`, NEVER `is not True` (the latter passes
# silently on None -- the project-wide add_bos/chat_template bug).
# A PARTIAL SHARD SET IS A FAILURE, never a silent skip.
#
# OUTPUT NAMES are prefixed `A04_7B_keep10f2_NBR_` and were verified to collide
# with NOTHING on this disk -- in particular not with the `A04_7B_keep12f2_*`
# dirs that `.73` is writing concurrently.
#
# NODE: `.82` ONLY (8xH20, zwfy6, numpy 2.4.6). `.73` is running the keep12
# 11-ckpt trajectory, `.104` paperC Qwen3 heal, LOCAL/`.21` SparseForge #246 --
# all outside this dispatch's budget and NOT to be touched.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU MMLU_BS CB_BS ARM STEPS TAG_PREFIX
#      KEEP_FRONT N_FRESH
# usage:
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/active/A04-recovery-certification/code/a04_keep10_neighbour_range_driver.sh \
#     > logs/a04_keep10_nbr_82.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-1124-7B}"
NGPU="${NGPU:-8}"
MMLU_BS="${MMLU_BS:-16}"      # frozen, matches the anchor + every A04 7B scan
CB_BS="${CB_BS:-32}"          # frozen, matches the anchor + every A04 7B scan
ARM="${ARM:-olmo2_probe2_7B_keep10fresh2}"
STEPS="${STEPS:-89000 89500 90000}"
TAG_PREFIX="${TAG_PREFIX:-A04_7B_keep10f2_NBR}"
KEEP_FRONT="${KEEP_FRONT:-10}"
N_FRESH="${N_FRESH:-2}"
CKDIR="outputs/$ARM"
PROG="logs/a04_keep10nbr_progress.log"

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs olmo2_mmlu_content_results olmo2_closedbook_results

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- refuse to share the node -----------------------------------------------
# The dispatch's hard budget is `.82` alone. If ANY other process holds GPU
# memory here, this is either the wrong node or a collision -- do not proceed.
gpu_free_or_die() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [ "$used" -gt 8000 ]; then
    note "REFUSE: ${used}MiB of GPU memory held by another process -- not sharing the node"
    exit 8
  fi
}
gpu_free_or_die

# --- refuse to be on the wrong node -----------------------------------------
# `.73` is writing A04_7B_keep12f2_* right now. Running there would contend for
# GPUs AND make the numpy of record 2.5.1 instead of the 2.4.6 the keep8
# comparison was published on. Both are silent failures, so both are checked.
#
# ! `hostname -I` on these nodes returns TEN addresses and 28.82.250.82 is NOT
# the first one (measured: `28.86.53.217 28.86.81.221 ... 28.82.250.82 ...`), so
# an `awk '{print $1}'` guard would REFUSE on the correct node. Match the whole
# list. Belt-and-braces: numpy must be 2.4.6, which is `.82`-specific in this
# cluster (.73/.104/.21 are 2.5.1, LOCAL is 2.3.5) -- so if the IP list ever
# changes shape, the numpy check still pins the node of record.
ALLIP=$(hostname -I 2>/dev/null)
NPV=$($PY -c 'import numpy;print(numpy.__version__)' 2>/dev/null)
if [ "${ALLOW_ANY_NODE:-0}" != "1" ]; then
  case " $ALLIP " in
    *" 28.82.250.82 "*) : ;;
    *) note "REFUSE: budgeted for .82 only; hostname -I = '$ALLIP'. \
Set ALLOW_ANY_NODE=1 only with an explicit new budget."; exit 7 ;;
  esac
  if [ "$NPV" != "2.4.6" ]; then
    note "REFUSE: numpy=$NPV but the keep8 comparison was published on .82/2.4.6; \
Generator.multinomial differs in 19/10000 rows across these versions."
    exit 7
  fi
fi
note "node ip_list='$ALLIP' numpy=$NPV"

run_mmlu() {  # $1=output_name $2=ckpt $3=step
  local NAME="$1" CK="$2" RD="olmo2_mmlu_content_results/$1"
  if [ -f "$RD/summary.json" ]; then note "mmlu SKIP $NAME (summary.json exists)"; return 0; fi
  note "mmlu START $NAME bs=$MMLU_BS"
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data --content_desc full \
    > "logs/a04k10nbr_mmlu_${NAME}_prepare.log" 2>&1 || true
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --content_desc full --num_shards $NGPU --shard_index $g \
      --batch_size $MMLU_BS --add_bos 0 \
      --output_name "$NAME" > "logs/a04k10nbr_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
  note "mmlu shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "mmlu ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" --n_boot 10000 \
      >> "logs/a04k10nbr_mmlu_${NAME}_merge.log" 2>&1
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
      --output_name "$NAME" > "logs/a04k10nbr_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "closedbook shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "closedbook ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04k10nbr_cb_${NAME}_merge.log" 2>&1
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
      --output_name "$NAME" > "logs/a04k10nbr_nq_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "nq_open shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "nq_open ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04k10nbr_nq_${NAME}_merge.log" 2>&1
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
  CK="$CKDIR/step${S}.pt"
  if [ ! -f "$CK" ]; then note "[skip] step${S}: no ckpt at $CK"; rc=1; continue; fi
  # torch.load probe. Byte SIZE IS NOT IDENTITY (all three of this triple share
  # 39,009,621,855 B), and one file in this repo (shortgpt16/step128000.pt) is a
  # truncated zip that `ls -l` cannot distinguish from healthy. So the meta
  # step/arch are asserted from the LOADED object before 8 GPUs are spent, and
  # the load itself exercises the zip central directory.
  probe=$($PY - "$CK" "$S" "$KEEP_FRONT" "$N_FRESH" <<'EOF' 2>/dev/null || echo bad
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
assert int(ck["step"]) == int(sys.argv[2]), f"meta step {ck['step']} != requested {sys.argv[2]}"
assert int(ck["keep_front_layers"]) == int(sys.argv[3]), ck["keep_front_layers"]
assert int(ck["n_fresh_layers"]) == int(sys.argv[4]), ck["n_fresh_layers"]
assert int(ck["num_hidden_layers"]) == int(sys.argv[3]) + int(sys.argv[4])
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
