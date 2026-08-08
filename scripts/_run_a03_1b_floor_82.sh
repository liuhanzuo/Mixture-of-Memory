#!/usr/bin/env bash
# ============================================================================
# A03 kill-condition test: "1B pilot all knowledge metrics sit at floor".
#
# Runs the OLMo-2 1B arms (intact 16L base / pruned+healed keep7+fresh2 / the
# barely-healed step500 lower bound) over the three knowledge axes A03 names
# for "old parametric knowledge": MMLU-content, TriviaQA, PopQA.
#
# Reuses the EXISTING harnesses verbatim (no new eval code, no arch drift):
#   scripts/eval_olmo2_mmlu_content.py   -> MMLU letter + content-text (dual)
#   scripts/eval_olmo2_closedbook_qa.py  -> PopQA + TriviaQA greedy closed-book
# Base protocol for both: chat_template=False, add_special_tokens=False
# (add_bos 0) -- OLMo-2 1B is a BASE LM with no SFT/RL.
#
# 8 shards over 8 GPUs per job; every job is hard-asserted for 8/8 shard files
# and for the exact expected item count before its merge is trusted.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU MMLU_BS CB_BS
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-0425-1B}"
NGPU="${NGPU:-8}"
MMLU_BS="${MMLU_BS:-64}"
CB_BS="${CB_BS:-48}"
CKDIR16=outputs/olmo2_probe2_1B_keep7fresh2_16card
CKDIR_E=outputs/olmo2_probe2_1B_keep7fresh2
PROG=logs/a03_1b_floor_progress.log

# cache is pre-verified present; stay fully offline so no job can hang on network
export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs olmo2_mmlu_content_results olmo2_closedbook_results

note() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" >> "$PROG"; }

run_mmlu() {  # $1=name $2=ckarg
  local NAME="$1" CKARG="$2" RD="olmo2_mmlu_content_results/$1"
  note "mmlu START arm=$NAME"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" $CKARG \
      --content_desc full --num_shards $NGPU --shard_index $g \
      --batch_size $MMLU_BS --add_bos 0 \
      --output_name "$NAME" > "logs/a03_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
  note "mmlu shards arm=$NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then note "mmlu ABORT arm=$NAME incomplete $ns/$NGPU"; return 9; fi
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" --n_boot 10000 \
      >> "logs/a03_mmlu_${NAME}_merge.log" 2>&1
  $PY - "$RD/summary.json" <<'EOF' || return 9
import json,sys
s=json.load(open(sys.argv[1])); exp=14042
tot=s["n"]
assert tot==exp, f"MMLU n={tot} != expected {exp}"
assert s["n_valid"]+s["n_nan"]==exp
print(f"OK n={tot} valid={s['n_valid']} nan={s['n_nan']} "
      f"letter={s['letter_acc']:.4f} content_norm={s['content_norm_acc']:.4f}")
EOF
  note "mmlu DONE arm=$NAME $($PY -c "
import json;s=json.load(open('$RD/summary.json'))
print(f\"letter={s['letter_acc']:.4f} content_norm={s['content_norm_acc']:.4f}\")")"
}

run_cb() {  # $1=name $2=ckarg
  local NAME="$1" CKARG="$2" RD="olmo2_closedbook_results/$1"
  note "closedbook START arm=$NAME"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" $CKARG \
      --tasks popqa,triviaqa --num_shards $NGPU --shard_index $g \
      --batch_size $CB_BS --add_bos 0 --max_new_tokens 32 \
      --output_name "$NAME" > "logs/a03_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "closedbook shards arm=$NAME -> $ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then note "closedbook ABORT arm=$NAME incomplete $ns/$NGPU"; return 9; fi
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" \
      >> "logs/a03_cb_${NAME}_merge.log" 2>&1
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

note "DRIVER START on $(hostname) ngpu=$NGPU mmlu_bs=$MMLU_BS cb_bs=$CB_BS"
rc=0
# each entry is "name|ckpt-args" (ckpt-args may contain a space -> quoted list)
for pair in "A03_1B_base|" \
            "A03_1B_keep7_step200k|--ckpt $CKDIR16/step200000.pt" \
            "A03_1B_keep7_step500|--ckpt $CKDIR_E/step500.pt"; do
  name="${pair%%|*}"; ck="${pair#*|}"
  run_mmlu "$name" "$ck" || rc=1
  run_cb   "$name" "$ck" || rc=1
done

note "DRIVER END rc=$rc"
exit $rc
