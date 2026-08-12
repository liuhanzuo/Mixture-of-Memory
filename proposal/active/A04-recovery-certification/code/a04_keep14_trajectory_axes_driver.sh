#!/usr/bin/env bash
# ============================================================================
# A04 — 4-axis capability scoring of the INTERMEDIATE keep14+fresh2 7B ckpts.
#
# WHY THIS EXISTS
# ---------------
# `STATUS.json:shallow_rung_ni_discrimination_20260812
#  .implication_for_pilot_two.cheap_next_steps_dominate[1]` asks for the
# intermediate 7B checkpoints on the keep14fresh2 and full32_dolmino
# trajectories to be scored, so the NI margin can be read as a CURVE in heal
# step rather than as a single endpoint verdict. That is what "the gate
# discriminates" would actually mean, and it costs zero training.
#
# The full32 half is NOT done here: `full32_rescore_v2_20260812
# .trajectory_scan_NOT_run` records that its four intermediate ckpts
# (step5000/10000/15000/20000, 81.6 GiB each) exist ONLY on wzc1, at a measured
# 16 MiB/s cross-disk rate = ~89 min per ckpt before any GPU work. This driver
# therefore covers exactly the part that is already resident on zwfy6:
#
#     outputs/olmo2_probe2_7B_keep14fresh2/step128000.pt        (48724473850 B)
#     outputs/olmo2_probe2_7B_keep14fresh2/keep14_step153500.pt (48724473850 B)
#
# Those two files have the SAME byte size (both carry optimizer_state; the
# 200k endpoint does not, hence its 16.2 GiB). Same size is NOT same weights:
# verified 2026-08-13 that head-400MiB and mid-200MiB-at-20GiB md5s all differ
# (29474e2b/dfe33568 vs ec9ed051/2c6eb15a), and torch.load reports step=128000
# vs step=153500 with keep_front=14 n_fresh=2 in both metas.
#
# PROTOCOL — FROZEN, RECOVERED FROM THE ARCHIVE'S OWN LOGS
# --------------------------------------------------------
# Not from prose. `logs/cb_driver_73.out` line 1/9 and
# `logs/nqopen_driver_73.log` line 2 echo the parameter set that produced the
# anchor (`base_full`, `base_full_nqopen`) and the step200k endpoint
# (`keep14_step200k`, `keep14_step200k_nqopen`):
#
#     closed-book:  scripts/_run_closedbook_8shard.sh, bs=32, add_bos=0,
#                   num_shards=8, max_new_tokens=32, max_ctx_len=512 (default),
#                   greedy (do_sample=False, num_beams=1), base LM, mode=pruned
#     MMLU-content: scripts/_run_olmo2_mmlu_content.sh via
#                   scripts/p06_run_transferred.sh, which sets NO `BS`, so the
#                   driver default `BS=16` is what produced 7B_base and
#                   7B_keep14_step200000. `git log -p --follow` on that file
#                   shows exactly ONE revision of the `BS=` line (`BS="${BS:-16}"`,
#                   commit d2e28f2), so 16 is not a later drift.
#                   --content_desc full --add_bos 0 --n_boot 10000.
#
# BATCH SIZE IS NOT A FREE PARAMETER. `full32_rescore_v2_20260812
# .sensitivity_bs48_probe` measured that bs32 -> bs48 flips 12/14267 popqa and
# 10/3610 nq_open items (bf16 left-pad-width numerics). So CB_BS must be 32 and
# MMLU_BS must be 16, matching the anchor and the endpoint this trajectory is
# read against. Any other value produces a number that cannot be differenced
# against the archive.
#
# chat_template=False: never set, because neither harness has a chat-template
# code path at all (`grep -n chat_template` matches only the docstring). These
# are BASE LMs with no SFT/RL. add_bos=0 is the same base protocol.
#
# ARCH: keep/fresh are read FROM THE CKPT META by `load_pruned_model`; a CLI
# mismatch raises. Passing --keep_front_layers 14 --n_fresh_layers 2 explicitly
# is therefore a free assertion that the file really is keep14+fresh2.
#
# HARNESS PARITY: scripts/eval_olmo2_closedbook_qa.py md5
# 2ed41993241226c795a3ca38375933f7 and scripts/eval_olmo2_mmlu_content.py md5
# fe4a62dbdf884a1e2aedc6ed26887b4e are IDENTICAL on wzc1 and zwfy6 and identical
# to the copies that produced the archive (recorded in
# `full32_rescore_v2_20260812.protocol_recovered` and
# `a04_step100k_axes_driver.sh` respectively) => same-CODE comparison, not
# code-version drift. The repo's standing rule is that same-arch/same-harness
# re-runs are BYTE-IDENTICAL, so a code delta would have made these cells
# non-comparable rather than merely noisy.
#
# NODE: zwfy6 only (.73/.82). Both have the cais/mmlu + PopQA + TriviaQA +
# nq_open caches (node-local /root/.cache and project data/hf_datasets_cache,
# same revision hashes on both), so both run fully offline. LOCAL/.21 lack the
# ckpts (wzc1 has its own copies but is running SparseForge) and .104 is running
# paperC Qwen3 heal -- both outside this dispatch's GPU budget.
#
# A PRESENT-BUT-PARTIAL SHARD SET IS A FAILURE, NEVER A SILENT SKIP: the repo
# has been corrupted before by a silently merged 5-of-8 set, so an incomplete
# directory is REMOVED rather than left for a later analysis to merge short.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU MMLU_BS CB_BS STEPS
# usage (on .73, step128000):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python STEPS=128000 \
#   setsid nohup bash proposal/active/A04-recovery-certification/code/a04_keep14_trajectory_axes_driver.sh \
#     > logs/a04_keep14_traj_128000.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-1124-7B}"
NGPU="${NGPU:-8}"
MMLU_BS="${MMLU_BS:-16}"      # frozen: p06_run_transferred.sh left BS unset -> 16
CB_BS="${CB_BS:-32}"          # frozen: cb_driver_73.out echoes bs=32
STEPS="${STEPS:-128000 153500}"
CKDIR=outputs/olmo2_probe2_7B_keep14fresh2
KEEP_FRONT=14
N_FRESH=2
PROG="logs/a04_keep14_traj_progress_$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d ' ').log"

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

# resolve the ckpt filename for a step: this arm's files are NOT uniformly named
# (step128000.pt but keep14_step153500.pt), so try both rather than guessing.
resolve_ckpt() {
  local s="$1"
  if   [ -f "$CKDIR/step${s}.pt" ];         then echo "$CKDIR/step${s}.pt"
  elif [ -f "$CKDIR/keep14_step${s}.pt" ];  then echo "$CKDIR/keep14_step${s}.pt"
  else echo ""; fi
}

run_mmlu() {  # $1=output_name $2=ckpt
  local NAME="$1" CK="$2" RD="olmo2_mmlu_content_results/$1"
  if [ -f "$RD/summary.json" ]; then note "mmlu SKIP $NAME (summary.json exists)"; return 0; fi
  note "mmlu START $NAME bs=$MMLU_BS"
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data --content_desc full \
    > "logs/a04k14_mmlu_${NAME}_prepare.log" 2>&1 || true
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --content_desc full --num_shards $NGPU --shard_index $g \
      --batch_size $MMLU_BS --add_bos 0 \
      --output_name "$NAME" > "logs/a04k14_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
  note "mmlu shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "mmlu ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" --n_boot 10000 \
      >> "logs/a04k14_mmlu_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1])); exp=14042
assert s["n"]==exp, f"MMLU n={s['n']} != expected {exp}"
assert s["n_valid"]+s["n_nan"]==exp, "n_valid+n_nan != n"
assert s["n_nan"]==0, f"n_nan={s['n_nan']} -- paired analysis needs an identical valid item set"
assert s["n_shards"]==8, f"n_shards={s['n_shards']}"
m=s["meta"]
assert m["mode"]=="pruned", m
assert m["keep_front_layers"]==14 and m["n_fresh_layers"]==2, m
assert m["num_hidden_layers"]==16, m
# add_bos must be FALSE. Written as `is False`, NOT `is not True`: the latter
# passes silently on None, which is the project-wide chat_template/add_bos
# assertion bug. KeyError if absent is the desired loud failure.
assert m["add_bos"] is False, f"add_bos={m['add_bos']!r} -- base protocol requires False"
assert m["content_desc"]=="full", m
print(f"OK n={s['n']} valid={s['n_valid']} nan={s['n_nan']} step={m['ckpt_step']} "
      f"letter={s['letter_acc']:.6f} content_norm={s['content_norm_acc']:.6f}")
EOF
  note "mmlu DONE $NAME"
}

run_cb() {  # $1=output_name $2=ckpt  (popqa+triviaqa in ONE dir, mirroring keep14_step200k)
  local NAME="$1" CK="$2" RD="olmo2_closedbook_results/$1"
  if [ -f "$RD/summary.json" ]; then note "closedbook SKIP $NAME (summary.json exists)"; return 0; fi
  note "closedbook START $NAME bs=$CB_BS"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --tasks popqa,triviaqa --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a04k14_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "closedbook shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "closedbook ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04k14_cb_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1]))
exp={"popqa":14267,"triviaqa":17944}
assert s["n_shards"]==8, s["n_shards"]
m=s["meta"]
assert m["mode"]=="pruned" and m["keep_front_layers"]==14 and m["n_fresh_layers"]==2, m
assert m["num_hidden_layers"]==16, m
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

run_nq() {  # $1=base_output_name $2=ckpt -- SEPARATE `_nqopen` dir, mirroring keep14_step200k_nqopen
  local NAME="${1}_nqopen" CK="$2"; local RD="olmo2_closedbook_results/$NAME"
  if [ -f "$RD/summary.json" ]; then note "nq_open SKIP $NAME (summary.json exists)"; return 0; fi
  note "nq_open START $NAME bs=$CB_BS"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" --ckpt "$CK" \
      --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH \
      --tasks nq_open --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a04k14_nq_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "nq_open shards $NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "nq_open ABORT $NAME incomplete $ns/$NGPU -- removing partial dir"
    rm -rf "$RD"; return 9
  fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a04k14_nq_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1]))
assert s["n_shards"]==8, s["n_shards"]
m=s["meta"]
assert m["mode"]=="pruned" and m["keep_front_layers"]==14 and m["n_fresh_layers"]==2, m
assert m["add_bos"] is False, f"add_bos={m['add_bos']!r} -- base protocol requires False"
assert m["max_new_tokens"]==32, m
v=s["tasks"]["nq_open"]
assert not v.get("skipped"), f"nq_open skipped: {v.get('error')}"
assert v["n"]==3610, f"nq_open n={v['n']} != expected 3610"
print(f"OK nq_open n={v['n']} em={v['em']:.8f} contains={v['contains']:.6f} maj_em={v['majority_em']:.6f}")
EOF
  note "nq_open DONE $NAME"
}

note "DRIVER START on $(hostname) ngpu=$NGPU mmlu_bs=$MMLU_BS cb_bs=$CB_BS steps='$STEPS'"
rc=0
for S in $STEPS; do
  CK="$(resolve_ckpt "$S")"
  if [ -z "$CK" ]; then note "[skip] step${S}: no ckpt in $CKDIR (tried step${S}.pt, keep14_step${S}.pt)"; rc=1; continue; fi
  # torch.load probe: independent of any size heuristic (both files are the SAME
  # size), so a partial write is bounced before 8 GPUs are spent on it, and the
  # step recorded in the meta is asserted to be the step we asked for.
  probe=$($PY - "$CK" "$S" <<'EOF' 2>/dev/null || echo bad
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(ck["step"]) == int(sys.argv[2]), f"meta step {ck['step']} != requested {sys.argv[2]}"
assert int(ck["keep_front_layers"]) == 14 and int(ck["n_fresh_layers"]) == 2, "not keep14+fresh2"
assert len(ck["model_state"]) == 179, len(ck["model_state"])
print("ok")
EOF
)
  if [ "$probe" != "ok" ]; then note "[skip] step${S}: ckpt probe failed for $CK"; rc=1; continue; fi
  note "ckpt OK step=$S -> $CK"
  gpu_free_or_die
  TAG="A04_7B_keep14f2_step${S}"
  t0=$(date +%s)
  run_cb   "$TAG" "$CK" || rc=1
  run_nq   "$TAG" "$CK" || rc=1
  run_mmlu "$TAG" "$CK" || rc=1
  note "step${S} ALL 4 AXES DONE in $(( $(date +%s) - t0 ))s"
done
note "DRIVER END rc=$rc"
exit $rc
