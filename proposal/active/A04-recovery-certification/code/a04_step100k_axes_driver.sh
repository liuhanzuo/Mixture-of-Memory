#!/usr/bin/env bash
# ============================================================================
# A04 — capability scoring at the REPAIRED PLATEAU rule's own earliest accept
# checkpoint.
#
# WHY THIS EXISTS
# ---------------
# `A04_PLATEAU_REPAIR_AND_MARGIN_SENSITIVITY.md` §1.4 records that the repaired
# rule R3 (`rate_5k = 100*(1-(ppl_c/ppl_prev)**(5000/d)) < T`, T=2.0 %/5k) FIRST
# ACCEPTS at step 100 000 (`rate_5k = 0.86012 %/5k`), not step 200 000 — while
# Pilot Zero scored capability axes ONLY at step 200 000. So the PLATEAU-vs-NI
# cell at the rule's own earliest accept point is UNMEASURED, not resolved, and
# there is 4.6386 % further relative PPL improvement between 100k and 200k. §5
# item 1 names this as GPU work that "is not done here". This driver does it.
#
# PROTOCOL — DELIBERATELY IDENTICAL TO THE step200k CELLS
# -------------------------------------------------------
# The step200k cells were produced by `scripts/_run_a03_1b_floor_82.sh` (MMLU +
# popqa/triviaqa) and `scripts/_run_a03_axes_floor_82.sh` (nq_open), both on a
# zwfy6 H20 node. Recovered from `logs/a03_1b_floor_progress.log` and
# `logs/a03_axes_floor_progress.log`:
#
#     ngpu=8  mmlu_bs=64  cb_bs=48  nq_bs=48
#     --content_desc full  --add_bos 0  --max_new_tokens 32  --merge --n_boot 10000
#
# Those exact values are reproduced below. `--add_bos 0` is the base-LM protocol
# (chat_template=False, no BOS) — OLMo-2-0425-1B has no SFT/RL, so a chat
# template would be unfair AND would break comparability with every existing
# cell. This is a project-wide rule, not a local choice.
#
# The two harness scripts are byte-identical across wzc1 and zwfy6 (md5
# fe4a62db… / 2ed41993…, verified 2026-08-12), and the only harness commit
# between the step200k eval (2026-08-08 20:51) and now is `7ac9653`
# (2026-08-08 22:29). `git show 7ac9653 -- scripts/eval_olmo2_mmlu_content.py`
# touches ONLY the `--any_family` base-mode routing branch (adds the
# `load_truncated_any_family` no-heal truncation path). The OLMo `--ckpt` ->
# `load_pruned_model` path this driver takes is UNCHANGED, so step100000 is
# scored by the same code that scored step200000. That matters because the
# repo's standing rule is that same-arch/same-harness re-runs are
# BYTE-IDENTICAL, so a code delta on the live path would have made the new cell
# non-comparable rather than merely noisy.
#
# ORDER: step100000 FIRST and complete, then 150000, then 50000. step100000 is
# the cell that closes the documented defect; the other two only turn one point
# into a bracket around the accept boundary. A partial run must still deliver
# the point that matters.
#
# ARCH: keep/fresh are read FROM THE CKPT META by `load_pruned_model`, and the
# CLI values must AGREE or it raises. Passing them explicitly is therefore a
# free assertion that the ckpt really is keep7+fresh2, not a guess.
#
# NODE: zwfy6 only (.73/.82/.104). The HF dataset caches (cais/mmlu, PopQA,
# TriviaQA, nq_open) live on zwfy6; on wzc1 this driver would fail offline.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU MMLU_BS CB_BS NQ_BS STEPS
# usage:
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/active/A04-recovery-certification/code/a04_step100k_axes_driver.sh \
#     > logs/a04_step100k_axes.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-0425-1B}"
NGPU="${NGPU:-8}"
MMLU_BS="${MMLU_BS:-64}"
CB_BS="${CB_BS:-48}"
NQ_BS="${NQ_BS:-48}"
# step100000 first: it is the cell that closes the defect.
STEPS="${STEPS:-100000 150000 50000}"
CKDIR=outputs/olmo2_probe2_1B_keep7fresh2_16card
PROG=logs/a04_step100k_axes_progress.log

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs olmo2_mmlu_content_results olmo2_closedbook_results

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- refuse to share the node -----------------------------------------------
# Checked once up front AND again per step: this driver is long enough that
# another agent could land on the node mid-run, and silently sharing 8 GPUs
# produces OOM-truncated shards rather than a clean failure.
gpu_free_or_die() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [ "$used" -gt 8000 ]; then
    note "REFUSE: ${used}MiB of GPU memory held by another process -- not sharing the node"
    exit 8
  fi
}
gpu_free_or_die

run_mmlu() {  # $1=name $2=ckpt
  local NAME="$1" CK="$2" RD="olmo2_mmlu_content_results/$1"
  if [ -f "$RD/summary.json" ]; then note "mmlu SKIP arm=$NAME (summary.json exists)"; return 0; fi
  note "mmlu START arm=$NAME"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers 7 --n_fresh_layers 2 \
      --content_desc full --num_shards $NGPU --shard_index $g \
      --batch_size $MMLU_BS --add_bos 0 \
      --output_name "$NAME" > "logs/a04_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
  note "mmlu shards arm=$NAME -> $ns/$NGPU"
  # A present-but-partial set is a FAILURE, never a silent skip: remove it so a
  # later analysis cannot merge 5-of-8 and report a short n as a real result.
  if [ "$ns" -ne "$NGPU" ]; then
    note "mmlu ABORT arm=$NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" --n_boot 10000 \
      >> "logs/a04_mmlu_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1])); exp=14042
assert s["n"]==exp, f"MMLU n={s['n']} != expected {exp}"
assert s["n_valid"]+s["n_nan"]==exp, "n_valid+n_nan != n"
assert s["n_nan"]==0, f"n_nan={s['n_nan']} -- paired analysis needs an identical valid item set"
assert s["n_shards"]==8, f"n_shards={s['n_shards']}"
print(f"OK n={s['n']} valid={s['n_valid']} nan={s['n_nan']} "
      f"letter={s['letter_acc']:.4f} content_norm={s['content_norm_acc']:.4f}")
EOF
  note "mmlu DONE arm=$NAME $($PY -c "
import json;s=json.load(open('$RD/summary.json'))
print(f\"letter={s['letter_acc']:.4f} content_raw={s['content_raw_acc']:.4f} content_norm={s['content_norm_acc']:.4f}\")")"
}

run_cb() {  # $1=name $2=ckpt   (popqa+triviaqa, same dir -- mirrors the baseline)
  local NAME="$1" CK="$2" RD="olmo2_closedbook_results/$1"
  if [ -f "$RD/summary.json" ]; then note "closedbook SKIP arm=$NAME (summary.json exists)"; return 0; fi
  note "closedbook START arm=$NAME"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers 7 --n_fresh_layers 2 \
      --tasks popqa,triviaqa --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a04_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "closedbook shards arm=$NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "closedbook ABORT arm=$NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04_cb_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1]))
exp={"popqa":14267,"triviaqa":17944}
assert s["n_shards"]==8, s["n_shards"]
for t,e in exp.items():
    v=s["tasks"][t]
    assert not v.get("skipped"), f"{t} skipped: {v.get('error')}"
    assert v["n"]==e, f"{t} n={v['n']} != expected {e}"
    print(f"OK {t} n={v['n']} em={v['em']:.4f} contains={v['contains']:.4f} maj_em={v['majority_em']:.4f}")
EOF
  note "closedbook DONE arm=$NAME"
}

run_nq() {  # $1=name $2=ckpt   -- SEPARATE dir suffixed _nq, mirroring the baseline
  local NAME="$1" CK="$2" NQRD="olmo2_closedbook_results/${1}_nq"
  if [ -f "$NQRD/summary.json" ]; then note "nq_open SKIP arm=$NAME (summary.json exists)"; return 0; fi
  note "nq_open START arm=$NAME"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers 7 --n_fresh_layers 2 \
      --tasks nq_open --num_shards $NGPU --shard_index $g \
      --batch_size $NQ_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "${NAME}_nq" > "logs/a04_nq_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$NQRD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "nq_open shards arm=$NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "nq_open ABORT arm=$NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$NQRD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "${NAME}_nq" \
      >> "logs/a04_nq_${NAME}_merge.log" 2>&1
  $PY - "$NQRD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1]))
assert s["n_shards"]==8, s["n_shards"]
v=s["tasks"]["nq_open"]
assert not v.get("skipped"), f"nq_open skipped: {v.get('error')}"
assert v["n"]==3610, f"nq_open n={v['n']} != expected 3610"
print(f"OK nq_open n={v['n']} em={v['em']:.4f} contains={v['contains']:.4f} maj_em={v['majority_em']:.4f}")
EOF
  note "nq_open DONE arm=$NAME"
}

note "DRIVER START on $(hostname) ngpu=$NGPU mmlu_bs=$MMLU_BS cb_bs=$CB_BS nq_bs=$NQ_BS steps='$STEPS'"
rc=0
for S in $STEPS; do
  CK="$CKDIR/step${S}.pt"
  if [ ! -f "$CK" ]; then note "[skip] step${S}: no ckpt at $CK"; rc=1; continue; fi
  # torch.load probe: independent of any size heuristic, so a partial write is
  # bounced before 8 GPUs are spent on it.
  hdr=$($PY -c "import torch; torch.load('$CK', map_location='cpu', weights_only=False); print('ok')" 2>/dev/null || echo bad)
  if [ "$hdr" != "ok" ]; then note "[skip] step${S}: ckpt not readable (partial write?)"; rc=1; continue; fi
  gpu_free_or_die
  TAG="A04_1B_keep7f2_step${S}"
  t0=$(date +%s)
  run_mmlu "$TAG" "$CK" || rc=1
  run_cb   "$TAG" "$CK" || rc=1
  run_nq   "$TAG" "$CK" || rc=1
  note "step${S} ALL 4 AXES DONE in $(( $(date +%s) - t0 ))s"
done
note "DRIVER END rc=$rc"
exit $rc
