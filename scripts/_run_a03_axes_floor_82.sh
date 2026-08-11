#!/usr/bin/env bash
# ============================================================================
# A03 additional axes floor gate: NQ-open only.
#
# This driver tests whether the OLMo-2 1B arms are above their construct-
# appropriate null floors on the NQ-open benchmark (validation, 3610 q),
# which serves as a proxy for A03's "multi-evidence / additional parametric
# knowledge" axis (the only additional axis with data on disk on zwfy6).
#
# WHY NQ-open (and not HotpotQA fullwiki or CounterFact):
#   - HotpotQA on disk is in LongBench format (54k-char context/document),
#     which is an OPEN-BOOK format, not a closed-book parametric test.
#   - CounterFact, MQuAKE, zsRE, KnowEdit (conflicting/updated knowledge):
#     NOT on either disk. These also require injected context (new facts can't
#     be measured closed-book from parametric knowledge) so even after download
#     they'd need new harness code.
#   - New injected facts: no standard closed-book benchmark exists by definition
#     (pre-training data does not contain post-cutoff facts).
#   - NQ-open IS on zwfy6 (data/hf_datasets_cache/google-research-datasets___nq_open,
#     6.7 MB, 3610 validation items) and IS fully implemented in the existing
#     eval_olmo2_closedbook_qa.py harness (lines 174-193, task='nq_open').
#
# AXIS COVERAGE:
#   NQ-open = closed-book, factoid, entity-centric -- a different parametric
#   knowledge distribution from TriviaQA/PopQA (more news-domain / short-form
#   Wikipedia). This helps A03 by:
#     (a) replicating the "old parametric knowledge" floor-cert on a 3rd dataset
#         to confirm the pattern is not TriviaQA/PopQA-specific;
#     (b) providing a moderately multi-hop-adjacent benchmark (some NQ questions
#         require implicit entity linking, though it is not formally multi-hop).
#
# WHAT IS NOT RUN HERE (and why):
#   - Conflicting/updated knowledge axis: no data on either disk AND no harness
#     for injecting new/conflicting facts into a base LM's context. Needs:
#     (1) CounterFact download (~20 MB via proxy + configs/password_hf_token.txt),
#     (2) new harness branch in eval_olmo2_closedbook_qa.py or a dedicated script
#         that injects "The [entity] is now [new value]." as a prompt prefix.
#   - New injected facts axis: requires CPT runs (the 6-arm design) before it is
#     testable; the floor gate for this axis must wait for the first training arm.
#   - True multi-hop (HotpotQA fullwiki closed-book): not on disk; fetchable via
#     proxy with HF token, but needs ~550 MB download + new harness branch.
#
# PROTOCOL (identical to _run_a03_1b_floor_82.sh):
#   - chat_template=False, add_special_tokens=False (--add_bos 0) -- BASE LM
#   - 8 shards over 8 GPUs (CUDA_VISIBLE_DEVICES 0..7)
#   - 8/8 shard files asserted present before merge
#   - exact item count (3610) asserted before merge is trusted
#   - greedy decode, max_new_tokens=32
#   - HF_HUB_OFFLINE=1 (data is pre-cached on zwfy6)
#
# env: PROJECT_ROOT PYTHON_BIN NGPU NQ_BS
#
# MUST RUN ON A ZWFY6 NODE (.82, .73, or .104) because NQ-open cache is on zwfy6.
#
# Usage (from .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash scripts/_run_a03_axes_floor_82.sh >logs/a03_axes_floor.out 2>&1 &
#
# Wall time estimate: ~15 min for 3 arms x 1 task on 8 H20 GPUs.
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-0425-1B}"
NGPU="${NGPU:-8}"
NQ_BS="${NQ_BS:-48}"
CKDIR16=outputs/olmo2_probe2_1B_keep7fresh2_16card
CKDIR_E=outputs/olmo2_probe2_1B_keep7fresh2
PROG=logs/a03_axes_floor_progress.log

# Stay fully offline; NQ-open cache must be present before running
export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs olmo2_closedbook_results

note() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" >> "$PROG"; }

# ---------------------------------------------------------------------------
# Pre-run assertions
# ---------------------------------------------------------------------------
# Assert that the NQ-open cache is present on this disk
NQ_CACHE="$W/data/hf_datasets_cache/google-research-datasets___nq_open"
if [ ! -d "$NQ_CACHE" ]; then
  echo "FATAL: NQ-open cache not found at $NQ_CACHE"
  echo "  This driver must run on a zwfy6 node (.82/.73/.104), not on wzc1."
  echo "  The NQ-open cache only exists on zwfy6 (6.7 MB)."
  exit 7
fi
note "NQ-open cache found at $NQ_CACHE"

# Assert that both checkpoint directories exist
if [ ! -f "$CKDIR16/step200000.pt" ]; then
  echo "FATAL: keep7f2 200k checkpoint not found: $CKDIR16/step200000.pt"
  exit 7
fi
if [ ! -f "$CKDIR_E/step500.pt" ]; then
  echo "FATAL: keep7f2 step500 checkpoint not found: $CKDIR_E/step500.pt"
  exit 7
fi
note "DRIVER START on $(hostname) ngpu=$NGPU nq_bs=$NQ_BS"
note "Arms: A03_1B_base (intact 16L) | A03_1B_keep7_step200k | A03_1B_keep7_step500"

# ---------------------------------------------------------------------------
# run_nq: run NQ-open for one arm
# $1=name, $2=ckpt-args (may be empty for base)
# ---------------------------------------------------------------------------
run_nq() {
  local NAME="$1" CKARG="$2" RD="olmo2_closedbook_results/$1"
  local EXPECTED_N=3610
  note "nq_open START arm=$NAME"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" $CKARG \
      --tasks nq_open --num_shards $NGPU --shard_index $g \
      --batch_size $NQ_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "${NAME}_nq" > "logs/a03_nq_${NAME}_shard${g}.log" 2>&1 &
  done
  wait

  # Assert shard completeness (8/8 required)
  local NQRD="olmo2_closedbook_results/${NAME}_nq"
  local ns; ns=$(ls "$NQRD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "nq_open shards arm=$NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "nq_open ABORT arm=$NAME: incomplete $ns/$NGPU shards (need exactly $NGPU)"
    echo "ABORT: arm=$NAME nq_open got $ns/$NGPU shards" >&2
    return 9
  fi

  # Merge shards
  $PY scripts/eval_olmo2_closedbook_qa.py --merge \
      --output_name "${NAME}_nq" \
      >> "logs/a03_nq_${NAME}_merge.log" 2>&1

  # Assert exact item count before trusting the merge
  $PY - "$NQRD/summary.json" <<EOF || return 9
import json,sys
s=json.load(open(sys.argv[1]))
exp={"nq_open": ${EXPECTED_N}}
assert s["n_shards"]==${NGPU}, f"n_shards={s['n_shards']} != expected ${NGPU}"
for t,e in exp.items():
    v=s["tasks"][t]
    assert not v.get("skipped"), f"{t} skipped: {v.get('error')}"
    assert v["n"]==e, f"{t} n={v['n']} != expected {e}"
    print(f"OK {t} n={v['n']} em={v['em']:.4f} contains={v['contains']:.4f} maj_em={v['majority_em']:.4f}")
EOF
  note "nq_open DONE arm=$NAME $($PY -c "
import json;s=json.load(open('$NQRD/summary.json'))
t=s['tasks']['nq_open']
print(f\"nq_open: em={t['em']:.4f} contains={t['contains']:.4f} maj_em={t['majority_em']:.4f}\")")"
}

# ---------------------------------------------------------------------------
# Run all three arms
# ---------------------------------------------------------------------------
rc=0
for pair in "A03_1B_base|" \
            "A03_1B_keep7_step200k|--ckpt $CKDIR16/step200000.pt" \
            "A03_1B_keep7_step500|--ckpt $CKDIR_E/step500.pt"; do
  name="${pair%%|*}"; ck="${pair#*|}"
  run_nq "$name" "$ck" || rc=1
done

note "DRIVER END rc=$rc"
if [ $rc -ne 0 ]; then
  echo "DRIVER FAILED rc=$rc -- check $PROG and individual shard logs"
  exit $rc
fi

echo "[$(date +%H:%M)] A03 axes floor gate DONE. NQ-open results:"
for name in A03_1B_base A03_1B_keep7_step200k A03_1B_keep7_step500; do
  RD="olmo2_closedbook_results/${name}_nq/summary.json"
  if [ -f "$RD" ]; then
    $PY -c "
import json
s=json.load(open('$RD'))
t=s['tasks']['nq_open']
print(f'  {\"$name\":35s}: em={t[\"em\"]:.4f} contains={t[\"contains\"]:.4f} maj_em={t[\"majority_em\"]:.4f} n={t[\"n\"]}')
" 2>/dev/null
  fi
done
echo ""
echo "Next step: run analyze_1b_knowledge_floor.py on the new arm results to get"
echo "floor-calibrated residuals (will require a new --tasks nq_open branch)."
echo "See: proposal/archive/A03-parametric-vs-external-memory/code/analyze_1b_knowledge_floor.py"
