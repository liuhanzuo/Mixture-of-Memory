#!/usr/bin/env bash
# ============================================================================
# A04 -- 4-axis capability scoring of the INTERMEDIATE full32_dolmino 7B ckpts.
#
# WHY THIS EXISTS
# ---------------
# `STATUS.json:shallow_rung_ni_discrimination_20260812
#  .implication_for_pilot_two.cheap_next_steps_dominate[1]` asks for the
# intermediate 7B ckpts on the keep14fresh2 AND full32_dolmino trajectories to
# be scored, so the NI margin can be read as a CURVE in step rather than as a
# single endpoint verdict. The keep14 half was done in commit 517c8d2
# (`a04_keep14_trajectory_axes_driver.sh`, all-REJECT). THIS driver does the
# full32 half, which is the load-bearing one:
#
#     full32_dolmino@step25000 is the ONLY NI ACCEPT in all of A04
#     (mmlu_content margin +1.0495 pp, 97.7% MMLU recovery), and its triviaqa
#     axis misses accepting by only 1.86 bootstrap SE.
#
# So the accept BOUNDARY, if A04 has one anywhere, is on this trajectory.
#
# ZERO STRUCTURAL DAMAGE -- READ THE RESULT ACCORDINGLY. full32 is
# `keep_front_layers=32 / n_fresh_layers=0`, i.e. all 32 pretrained layers
# present and nothing transplanted. It is a CONTINUED-PRETRAINING control, not a
# healed model. Any accept found here is a statement about CPT drift on the heal
# corpus, NOT about recovery from structural injury
# (`shallow_rung_ni_discrimination_20260812.the_load_bearing_new_finding.caveat`).
#
# WHY `mode=pruned` FOR AN UNPRUNED MODEL
# ---------------------------------------
# `load_pruned_model` labels every ckpt-loaded model `mode: "pruned"` regardless
# of shape, and the ARCHIVED step25000 dirs
# (`olmo2_closedbook_results/full32_step25000/summary.json:meta`) record exactly
# `mode=pruned keep_front_layers=32 n_fresh_layers=0 num_hidden_layers=32`.
# Verified on disk 2026-08-13. The archive's convention is reproduced verbatim
# here: changing it would make the new cells non-differenceable against the
# endpoint. `--keep_front_layers 32 --n_fresh_layers 0` is a free assertion,
# since `load_pruned_model` reads keep/fresh from the ckpt meta and RAISES on a
# CLI mismatch.
#
# PROTOCOL -- FROZEN, IDENTICAL TO THE ARCHIVE AND TO THE keep14 HALF
# -------------------------------------------------------------------
#     closed-book:  bs=32, add_bos=0, num_shards=8, max_new_tokens=32,
#                   greedy (do_sample=False, num_beams=1), base LM
#     MMLU-content: bs=16, --content_desc full, --add_bos 0, --n_boot 10000
#
# Recovered from the archive's OWN scheduler logs, not from prose:
# `full32_rescore_v2_20260812.protocol_recovered` reads bs=32 out of
# `logs/cb_full32_step25000{,_nqopen}_sched.out`; MMLU bs=16 is
# `_run_olmo2_mmlu_content.sh`'s default, which `p06_run_transferred.sh` left
# unset and which `git log -p --follow` shows has exactly ONE revision (d2e28f2).
#
# BATCH SIZE IS NOT FREE: `full32_rescore_v2_20260812.sensitivity_bs48_probe`
# measured bs32 -> bs48 flipping 12/14267 popqa and 10/3610 nq_open items (bf16
# left-pad-width numerics). Any other value yields a number that cannot be
# differenced against the archive.
#
# chat_template=False: never set, because NEITHER harness has a chat-template
# code path (the only occurrence of the string in either file is a docstring).
# These are BASE LMs with no SFT/RL. add_bos=0 is the same base protocol.
#
# HARNESS PARITY: `eval_olmo2_closedbook_qa.py` md5
# 2ed41993241226c795a3ca38375933f7 and `eval_olmo2_mmlu_content.py` md5
# fe4a62dbdf884a1e2aedc6ed26887b4e -- verified 2026-08-13 IDENTICAL on wzc1,
# .73 and .82, and identical to the copies that produced the archived endpoint
# and the anchor. Same-CODE comparison, not code-version drift.
#
# CKPTS ARE STAGED COPIES. The four intermediate ckpts are wzc1-resident; they
# were moved by `a04_full32_stage_parallel.sh` with FULL-FILE sha256 equality on
# both disks plus a zip-entry-count check (see that file's header for why a
# prefix hash is the wrong check when truncation is the failure mode). This
# driver additionally torch.load-probes each file and asserts the ckpt meta's
# step/keep/fresh/tensor-count BEFORE spending 8 GPUs.
#
# NODE: zwfy6 only (.73/.82). NOT touched: .104 (paperC Qwen3 heal), LOCAL/.21
# (SparseForge #246) -- wzc1 is read-only here, as the ckpt source.
#
# A PRESENT-BUT-PARTIAL SHARD SET IS A FAILURE, NEVER A SILENT SKIP: the repo
# has been corrupted before by a silently merged 5-of-8 set, so an incomplete
# directory is REMOVED rather than left for a later analysis to merge short.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU MMLU_BS CB_BS STEPS CKPT_DIR
# usage (on .73, steps 5000 and 10000):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python STEPS="5000 10000" \
#   setsid nohup bash proposal/active/A04-recovery-certification/code/a04_full32_trajectory_axes_driver.sh \
#     > logs/a04_full32_traj_73.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-1124-7B}"
NGPU="${NGPU:-8}"
MMLU_BS="${MMLU_BS:-16}"      # frozen: _run_olmo2_mmlu_content.sh default, produced 7B_base
CB_BS="${CB_BS:-32}"          # frozen: cb_full32_step25000_sched.out echoes bs=32
STEPS="${STEPS:-5000 10000 15000 20000}"
CKPT_DIR="${CKPT_DIR:-outputs/a04_staged}"
KEEP_FRONT=32
N_FRESH=0
EXPECT_TENSORS=355            # full32 is 32 layers -> 355 tensors (keep14f2 has 179)
PROG="logs/a04_full32_traj_progress_$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d ' ').log"

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs olmo2_mmlu_content_results olmo2_closedbook_results

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- refuse to share the node -----------------------------------------------
# Checked up front AND per step: silently sharing 8 GPUs produces OOM-truncated
# shards rather than a clean failure, and this driver runs long enough that
# another agent could land mid-run.
gpu_free_or_die() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [ "$used" -gt 8000 ]; then
    note "REFUSE: ${used}MiB of GPU memory held by another process -- not sharing the node"
    exit 8
  fi
}
gpu_free_or_die

resolve_ckpt() {  # staged name first, then a native copy if one ever appears
  local s="$1"
  if   [ -f "$CKPT_DIR/full32_step${s}_from_wzc1.pt" ]; then echo "$CKPT_DIR/full32_step${s}_from_wzc1.pt"
  elif [ -f "outputs/olmo2_probe2_7B_full32_dolmino/step${s}.pt" ]; then echo "outputs/olmo2_probe2_7B_full32_dolmino/step${s}.pt"
  else echo ""; fi
}

run_mmlu() {  # $1=output_name $2=ckpt
  local NAME="$1" CK="$2" RD="olmo2_mmlu_content_results/$1"
  if [ -f "$RD/summary.json" ]; then note "mmlu SKIP $NAME (summary.json exists)"; return 0; fi
  note "mmlu START $NAME bs=$MMLU_BS"
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data --content_desc full \
    > "logs/a04f32_mmlu_${NAME}_prepare.log" 2>&1 || true
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --content_desc full --num_shards $NGPU --shard_index $g \
      --batch_size $MMLU_BS --add_bos 0 \
      --output_name "$NAME" > "logs/a04f32_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
  note "mmlu shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "mmlu ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" --n_boot 10000 \
      >> "logs/a04f32_mmlu_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1])); exp=14042
assert s["n"]==exp, f"MMLU n={s['n']} != expected {exp}"
assert s["n_valid"]+s["n_nan"]==exp, "n_valid+n_nan != n"
assert s["n_nan"]==0, f"n_nan={s['n_nan']} -- paired analysis needs an identical valid item set"
assert s["n_shards"]==8, f"n_shards={s['n_shards']}"
m=s["meta"]
# mode=pruned is the ARCHIVE's own label for this arm (verified on disk in
# olmo2_closedbook_results/full32_step25000/summary.json), even though full32 is
# structurally undamaged: load_pruned_model labels any ckpt-loaded model this way.
assert m["mode"]=="pruned", m
assert m["keep_front_layers"]==32 and m["n_fresh_layers"]==0, m
assert m["num_hidden_layers"]==32, m
# add_bos must be FALSE. Written as `is False`, NOT `is not True`: the latter
# passes silently on None, which is the project-wide add_bos/chat_template
# assertion bug. KeyError if absent is the desired loud failure.
assert m["add_bos"] is False, f"add_bos={m['add_bos']!r} -- base protocol requires False"
assert m["content_desc"]=="full", m
print(f"OK n={s['n']} valid={s['n_valid']} nan={s['n_nan']} step={m['ckpt_step']} "
      f"letter={s['letter_acc']:.6f} content_norm={s['content_norm_acc']:.6f}")
EOF
  note "mmlu DONE $NAME"
}

run_cb() {  # $1=output_name $2=ckpt  (popqa+triviaqa in ONE dir, mirroring full32_step25000)
  local NAME="$1" CK="$2" RD="olmo2_closedbook_results/$1"
  if [ -f "$RD/summary.json" ]; then note "closedbook SKIP $NAME (summary.json exists)"; return 0; fi
  note "closedbook START $NAME bs=$CB_BS"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --tasks popqa,triviaqa --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a04f32_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "closedbook shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "closedbook ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04f32_cb_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1]))
exp={"popqa":14267,"triviaqa":17944}
assert s["n_shards"]==8, s["n_shards"]
m=s["meta"]
assert m["mode"]=="pruned" and m["keep_front_layers"]==32 and m["n_fresh_layers"]==0, m
assert m["num_hidden_layers"]==32, m
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

run_nq() {  # $1=base_output_name $2=ckpt -- SEPARATE `_nqopen` dir, mirroring full32_step25000_nqopen
  local NAME="${1}_nqopen" CK="$2"; local RD="olmo2_closedbook_results/$NAME"
  if [ -f "$RD/summary.json" ]; then note "nq_open SKIP $NAME (summary.json exists)"; return 0; fi
  note "nq_open START $NAME bs=$CB_BS"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --tasks nq_open --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a04f32_nq_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "nq_open shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "nq_open ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04f32_nq_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1]))
assert s["n_shards"]==8, s["n_shards"]
m=s["meta"]
assert m["mode"]=="pruned" and m["keep_front_layers"]==32 and m["n_fresh_layers"]==0, m
assert m["add_bos"] is False, f"add_bos={m['add_bos']!r} -- base protocol requires False"
assert m["max_new_tokens"]==32, m
v=s["tasks"]["nq_open"]
assert not v.get("skipped"), f"nq_open skipped: {v.get('error')}"
assert v["n"]==3610, f"nq_open n={v['n']} != expected 3610"
print(f"OK nq_open n={v['n']} em={v['em']:.8f} contains={v['contains']:.6f} maj_em={v['majority_em']:.6f}")
EOF
  note "nq_open DONE $NAME"
}

note "DRIVER START on $(hostname) ngpu=$NGPU mmlu_bs=$MMLU_BS cb_bs=$CB_BS steps='$STEPS' ckpt_dir=$CKPT_DIR"
rc=0
for S in $STEPS; do
  CK="$(resolve_ckpt "$S")"
  if [ -z "$CK" ]; then note "[skip] step${S}: no ckpt found for step $S"; rc=1; continue; fi
  # torch.load probe. Independent of any size heuristic: step10000/15000/20000/25000
  # are all EXACTLY 87583881932 B, so size can never identify which is which, and
  # the known cluster failure mode (shortgpt16/step128000.pt on zwfy6) is a file
  # that looks plausible to `ls -l` and fails to open. Bounce it before 8 GPUs.
  probe=$($PY - "$CK" "$S" "$EXPECT_TENSORS" <<'EOF' 2>/dev/null || echo bad
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
assert int(ck["step"]) == int(sys.argv[2]), f"meta step {ck['step']} != requested {sys.argv[2]}"
assert int(ck["keep_front_layers"]) == 32, ck["keep_front_layers"]
assert int(ck["n_fresh_layers"]) == 0, ck["n_fresh_layers"]
assert int(ck["num_hidden_layers"]) == 32, ck["num_hidden_layers"]
assert len(ck["model_state"]) == int(sys.argv[3]), len(ck["model_state"])
print("ok")
EOF
)
  if [ "$probe" != "ok" ]; then note "[skip] step${S}: ckpt probe failed for $CK"; rc=1; continue; fi
  note "ckpt OK step=$S -> $CK"
  gpu_free_or_die
  TAG="A04_7B_full32_step${S}"
  t0=$(date +%s)
  run_cb   "$TAG" "$CK" || rc=1
  run_nq   "$TAG" "$CK" || rc=1
  run_mmlu "$TAG" "$CK" || rc=1
  note "step${S} ALL 4 AXES DONE in $(( $(date +%s) - t0 ))s"
done
note "DRIVER END rc=$rc"
exit $rc
