#!/usr/bin/env bash
# B-P0.6: the DECISIVE missing cell for Paper B's positioning.
#
# ShortGPT-16 @step200000 is the strongest competing pruning baseline WE ran, and
# it BEATS our main arm on held-out Dolmino PPL (9.7800 vs keep14+fresh2 10.5613,
# both 8-shard/4096-window). Crucially the two are depth- and size-matched:
#   ShortGPT-16 : keep_front_layers=16, n_fresh_layers=0, 16L, n_params=4,060,352,512
#   keep14+fresh2: 14 inherited + 2 freshly random-init,  16L, ~same params
# so the PPL gap is attributable to WHICH layers survive (BI-metric non-contiguous
# selection vs contiguous keep-front + fresh graft), not to capacity.
#
# But ShortGPT-16 has capability numbers ONLY at step0 (olmo2_downstream_results/
# 7B_shortgpt_step0*). Without @200k core6+know5 we cannot say whether the PPL
# win transfers to knowledge/capability -- which is exactly Paper B's claim
# (perplexity heals while knowledge lags). If ShortGPT-16 also wins on capability,
# the "our construction is better" framing is dead; if it wins PPL but NOT
# capability, that is the single best piece of evidence Paper B has, because the
# dissociation would then be demonstrated ACROSS two different pruning schemes.
#
# Protocol is byte-identical to _run_olmo2_probe2_downstream_keep14_8gpu.sh
# (same harness, same TASKS split, BS=8, verbatim 8-shard [g::8] + merge,
# acc = sum(correct)/sum(n)); ONLY --ckpt differs. Base protocol per project rule:
# chat_template=False, no BOS, likelihood-based MC (OLMo-2 is a BASE LM).
#
# Runs on wzc1 (LOCAL or .252) because outputs/olmo2_probe2_7B_shortgpt16/
# step200000.pt lives on wzc1.
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/hf_datasets_cache
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

CKPT=outputs/olmo2_probe2_7B_shortgpt16/step200000.pt
BASE=../models/OLMo-2-1124-7B
BS=8
DONE=logs/olmo2_downstream_shortgpt200k_DONE
rm -f "$DONE"

if [ ! -f "$CKPT" ]; then
  echo "FATAL: ckpt missing: $CKPT" | tee "$DONE"; exit 1
fi

# Two legs, matching how keep14 was evaluated (core6 then know5), so the numbers
# land in directories directly comparable to 7B_keep14_step200000{,_know}.
LEGS=(
  "7B_shortgpt_step200000|hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
  "7B_shortgpt_step200000_know|mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"
)

for leg in "${LEGS[@]}"; do
  NAME="${leg%%|*}"; TASKS="${leg#*|}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] LEG $NAME tasks=$TASKS"
  echo "[$(date '+%F %T')] prepare_data"
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
    > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1
  tail -4 "logs/olmo2_downstream_${NAME}_prepare.log"

  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$TASKS" \
      --num_shards 8 --shard_index $g --batch_size $BS \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait

  # Assert all 8 shards produced output before merging. A silent partial merge
  # changes the denominator and yields a plausible-but-wrong accuracy -- exactly
  # the failure that corrupted two points of the #103 PPL curve on 2026-08-05.
  NSH=$(ls olmo2_downstream_results/"$NAME"/shard*of8.json 2>/dev/null | wc -l)
  if [ "$NSH" -ne 8 ]; then
    echo "[$(date '+%F %T')] FATAL $NAME: only $NSH/8 shards present -- NOT merging." | tee -a "$DONE"
    grep -lE 'Error|Traceback|OutOfMemory|CUBLAS' logs/olmo2_downstream_${NAME}_shard*.log 2>/dev/null | tee -a "$DONE"
    continue
  fi
  echo "[$(date '+%F %T')] $NAME 8/8 shards present; merging"
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
done

echo "[$(date '+%F %T')] SHORTGPT200K DOWNSTREAM DONE" | tee "$DONE"
for leg in "${LEGS[@]}"; do
  NAME="${leg%%|*}"
  echo "--- $NAME ---" >> "$DONE"
  cat "olmo2_downstream_results/${NAME}/summary.json" >> "$DONE" 2>/dev/null
done

# Side-by-side against our main arm, so the decisive comparison is in the log.
echo "" >> "$DONE"
echo "=== ShortGPT-16@200k  vs  keep14+fresh2@200k (both 16L, ~4.06B, same harness) ===" >> "$DONE"
$PY - >> "$DONE" 2>&1 <<'PYEOF'
import json, os
CORE6 = ["hellaswag","arc_challenge","arc_easy","piqa","winogrande","openbookqa"]
KNOW5 = ["mmlu","lambada_openai","boolq","commonsense_qa","social_iqa"]
def load(name):
    p = f"olmo2_downstream_results/{name}/summary.json"
    if not os.path.exists(p): return None
    d = json.load(open(p))
    t = d.get("tasks", d)
    out = {}
    for k, v in t.items():
        if isinstance(v, dict):
            a = v.get("acc", v.get("accuracy", v.get("em")))
            if a is not None: out[k] = a
    return out
pairs = [("ShortGPT-16", "7B_shortgpt_step200000", "7B_shortgpt_step200000_know"),
         ("keep14+fresh2", "7B_keep14_step200000", "7B_keep14_step200000_know")]
rows = {}
for label, c, k in pairs:
    m = {}
    for nm in (c, k):
        r = load(nm)
        if r: m.update(r)
    rows[label] = m
alltasks = CORE6 + KNOW5
print(f"{'task':18s} {'ShortGPT-16':>12s} {'keep14+fresh2':>14s} {'delta(SG-k14)':>14s}")
for t in alltasks:
    a = rows.get("ShortGPT-16", {}).get(t)
    b = rows.get("keep14+fresh2", {}).get(t)
    if a is None or b is None:
        print(f"{t:18s} {str(a):>12s} {str(b):>14s} {'n/a':>14s}"); continue
    print(f"{t:18s} {a:12.4f} {b:14.4f} {a-b:+14.4f}")
for grp, ts in (("core6", CORE6), ("know5", KNOW5)):
    va = [rows.get("ShortGPT-16", {}).get(t) for t in ts]
    vb = [rows.get("keep14+fresh2", {}).get(t) for t in ts]
    if all(x is not None for x in va+vb):
        ma, mb = sum(va)/len(va), sum(vb)/len(vb)
        print(f"{grp+' MEAN':18s} {ma:12.4f} {mb:14.4f} {ma-mb:+14.4f}")
print()
print("held-out Dolmino PPL (8-shard/4096win, lower=better):"
      " ShortGPT-16=9.7800  keep14+fresh2=10.5613  -> ShortGPT wins PPL by 0.78")
print("READ: if ShortGPT also wins core6/know5, the 'our construction is better' framing is dead.")
print("      if it wins PPL but NOT knowledge, that is Paper B's dissociation, shown across two")
print("      independent pruning schemes -- a STRONGER result than a single-arm claim.")
PYEOF
echo "[$(date '+%F %T')] wrote $DONE"
