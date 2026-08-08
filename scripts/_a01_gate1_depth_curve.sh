#!/usr/bin/env bash
# Generic A01 gate-1 damaged depth-curve driver.
# Env vars:
#   ROOT_DIR  -- project root on this node (wzc1 or zwfy6)
#   FAMILY    -- Llama-2-7B | Llama-3-8B | Qwen3-8B  (label only; MODEL_PATH is the truth)
#   MODEL_PATH -- relative or absolute path to the model dir
#   KEEPS     -- space-separated list of keep_front_layers to try
#   TAG       -- filename prefix, e.g. "gate1_dmg_llama3_8b_depth"
# Idempotent per (family, keep) via output-dir summary.json presence.
set -u

ROOT=${ROOT_DIR:?ROOT_DIR required}
FAMILY=${FAMILY:?FAMILY required}
MODEL=${MODEL_PATH:?MODEL_PATH required}
KEEPS=${KEEPS:?KEEPS required}
TAG=${TAG:?TAG required}

cd "$ROOT"
PY=${PY:-/opt/conda/envs/torch-base/bin/python}
NGPU=${NGPU:-8}
BS=${BS:-16}
EXPECT_N=14042
SCRIPT=scripts/eval_olmo2_mmlu_content.py
PROGRESS=logs/${TAG}_progress.log

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
mkdir -p logs olmo2_mmlu_content_results

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_arm() {
    local keep="$1"
    local arm="${TAG}_k${keep}"
    local out="olmo2_mmlu_content_results/${arm}"
    if [[ -f "${out}/summary.json" ]]; then
        log "SKIP arm=${arm} (summary.json exists)"; return 0
    fi
    log "arm=${arm} START model=${MODEL} keep=${keep}"

    local pids=()
    for i in $(seq 0 $((NGPU-1))); do
        CUDA_VISIBLE_DEVICES="$i" "$PY" "$SCRIPT" \
            --base_model "$MODEL" --any_family --keep_front_layers "$keep" \
            --output_name "$arm" \
            --num_shards "$NGPU" --shard_index "$i" \
            --batch_size "$BS" --add_bos 0 --content_desc full \
            > "logs/${arm}_shard${i}.log" 2>&1 &
        pids+=($!)
    done
    local fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    local nshard
    nshard=$(ls "${out}"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
    log "  arm=${arm} shards ${nshard}/${NGPU}"
    if (( fail > 0 || nshard != NGPU )); then
        log "  arm=${arm} INCOMPLETE -- NOT merging"; return 1
    fi
    "$PY" "$SCRIPT" --merge --output_name "$arm" --num_shards "$NGPU" \
        --base_model "$MODEL" >> "logs/${arm}_merge.log" 2>&1

    "$PY" - "$out" "$EXPECT_N" <<'PY' | tee -a "$PROGRESS"
import json, sys
out, expect = sys.argv[1], int(sys.argv[2])
d = json.load(open(f"{out}/summary.json"))
n = d.get("n", d.get("n_total"))
if n != expect:
    raise SystemExit(f"MERGE INTEGRITY FAIL {out}: n={n} != {expect}")
print(f"  MERGE OK {out}: n={n} letter={d.get('letter_acc'):.4f} "
      f"content_norm={d.get('content_norm_acc'):.4f} nan={d.get('n_nan')}")
PY
}

log "DRIVER START $(hostname) family=$FAMILY model=$MODEL keeps={$KEEPS}"
for k in $KEEPS; do run_arm "$k"; done
log "ALL DONE"
