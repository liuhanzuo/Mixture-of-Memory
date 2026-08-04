#!/usr/bin/env bash
# OOD (+ in-domain gap-fill) PPL task-pool for the OLMo-2 prune-then-heal probe.
# 8 GPU workers pop (model, corpus) jobs from a flock'd queue and run the SAME
# harness as the in-domain held-out PPL (scripts/eval_olmo2_probe2_ppl.py) so OOD
# and in-domain numbers are directly comparable. Each job is single-GPU
# (num_shards=1) then merged to summary.json.
#
# Usage:
#   PROJECT_ROOT=/abs/root PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   bash scripts/_run_ood_ppl_pool.sh
set -u

ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$ROOT"
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
BS="${BS:-8}"
OOD_ROOT="ood_ppl_results"
IND_ROOT="olmo2_ppl_results"
mkdir -p "$OOD_ROOT" logs
QUEUE="$(mktemp /tmp/ood_ppl_queue.XXXXXX)"
LOCK="${QUEUE}.lock"

WT="data/ood_ppl/wikitext103_test.npy"
PG="data/ood_ppl/pg19_test.npy"
IND="data/dolmino_now_val.npy"

# job = TAG|CKPT|VAL|OUTPUT_NAME|RESULTS_ROOT   (CKPT empty => base mode)
declare -a JOBS
add(){ JOBS+=("$1|$2|$3|$4|$5"); }

K_FULL32="outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt"
K_KEEP14="outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt"
K_SHORT="outputs/olmo2_probe2_7B_shortgpt16/step200000.pt"
K_RAND="outputs/olmo2_probe2_7B_keep14fresh2_fromscratch/step200000.pt"
K_FROZ="outputs/olmo2_probe2_7B_keep14fresh2_freezefront/step200000.pt"

# ---- OOD: 6 models x {wikitext, pg19} ----
for c in "wikitext103:$WT" "pg19:$PG"; do
  cn="${c%%:*}"; cp="${c##*:}"
  add "base_$cn"     ""          "$cp" "base_$cn"            "$OOD_ROOT"
  add "full32_$cn"   "$K_FULL32" "$cp" "full32_step25000_$cn" "$OOD_ROOT"
  add "keep14_$cn"   "$K_KEEP14" "$cp" "keep14_step200000_$cn" "$OOD_ROOT"
  add "shortgpt_$cn" "$K_SHORT"  "$cp" "shortgpt_step200000_$cn" "$OOD_ROOT"
  add "random_$cn"   "$K_RAND"   "$cp" "random_step200000_$cn" "$OOD_ROOT"
  add "frozen_$cn"   "$K_FROZ"   "$cp" "frozen_step200000_$cn" "$OOD_ROOT"
done
# ---- in-domain gap-fill (base + shortgpt@200k not previously computed) ----
add "base_indomain"     ""         "$IND" "7B_base_indomain"            "$IND_ROOT"
add "shortgpt_indomain" "$K_SHORT" "$IND" "7B_shortgpt_step200000_indomain" "$IND_ROOT"

printf '%s\n' "${JOBS[@]}" > "$QUEUE"
echo "[pool] $(wc -l < "$QUEUE") jobs queued -> $QUEUE"

pop(){ # atomic pop of the first queue line -> stdout (empty when drained)
  ( flock 9
    line="$(head -n1 "$QUEUE")"
    if [ -n "$line" ]; then sed -i '1d' "$QUEUE"; fi
    printf '%s' "$line"
  ) 9>"$LOCK"
}

worker(){
  local gpu="$1"
  while :; do
    local job; job="$(pop)"
    [ -z "$job" ] && break
    IFS='|' read -r tag ckpt val out root <<< "$job"
    local log="logs/ood_ppl_${out}.log"
    echo "[gpu$gpu] START $tag -> $root/$out  (log $log)"
    local ckarg=""; [ -n "$ckpt" ] && ckarg="--ckpt $ckpt"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_olmo2_probe2_ppl.py \
        --base_model "$BASE" $ckarg --val_path "$val" \
        --output_name "$out" --results_root "$root" \
        --num_shards 1 --shard_index 0 --batch_size "$BS" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_olmo2_probe2_ppl.py \
          --merge --output_name "$out" --results_root "$root" >> "$log" 2>&1
      local ppl; ppl="$($PY -c "import json;print(round(json.load(open('$root/$out/summary.json'))['ppl'],4))" 2>/dev/null)"
      echo "[gpu$gpu] DONE  $tag  PPL=$ppl"
    else
      echo "[gpu$gpu] FAIL  $tag rc=$rc (see $log)"
    fi
  done
  echo "[gpu$gpu] drained, exiting"
}

for g in 0 1 2 3 4 5 6 7; do worker "$g" & done
wait
echo "[pool] ALL DONE"
rm -f "$QUEUE" "$LOCK"
