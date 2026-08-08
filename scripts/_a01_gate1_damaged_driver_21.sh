#!/usr/bin/env bash
# A01 gate-1 extension: DAMAGED third-family arms (no heal).
#
# WHY THIS ADDS THE MISSING LEG OF THE GATE-1 KILL TEST
# -----------------------------------------------------
# The first gate-1 run on .21 showed that HEALTHY Llama-2 / Llama-3 / Qwen3-8B
# all keep the letter interface fully above their construct-appropriate floor.
# That answers "does the interface work on a non-OLMo INTACT model" (yes it does).
# It does NOT answer A01's actual load-bearing claim, which is that STRUCTURAL
# damage causes letter interface to degenerate below floor. On OLMo-2 that shows
# up as keep8 letter=0.2550 < always-D=0.2689 with tie rate 30.64%.
#
# So this driver runs the same MMLU letter-vs-content probe on the same three
# non-OLMo families AFTER truncating them to their first 8 (or 12) transformer
# blocks without heal training. Predictions:
#   * If damaged non-OLMo letter also drops to/below its floor with high tie
#     rate: letter-interface failure is a general property of damaged 7-8B LMs.
#     A01's OLMo-only scoping would be wrong; the phenomenon is a good target.
#   * If damaged non-OLMo letter STAYS above floor with modest tie rate: even
#     structural damage does not reproduce the OLMo pathology across families.
#     A01's OLMo-specific scoping is now supported from both directions
#     (healthy AND damaged), which is the strongest possible narrowing.
#
# The truncation is NO-HEAL: `model.model.layers = layers[:N]`, no fresh block.
# This is harsher than OLMo's healed keep-front + fresh setup and is a
# deliberately worst-case damaged probe.
set -u

ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
NGPU=8
BS=16
EXPECT_N=14042

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4

mkdir -p logs olmo2_mmlu_content_results

SCRIPT=scripts/eval_olmo2_mmlu_content.py
PROGRESS=logs/a01_gate1_damaged_progress.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_arm() {
    local arm="$1" model="$2" keep="$3"
    local out="olmo2_mmlu_content_results/${arm}"
    if [[ -f "${out}/summary.json" ]]; then
        log "SKIP arm=${arm} (summary.json exists)"; return 0
    fi
    if [[ ! -d "$model" ]]; then
        log "SKIP arm=${arm} -- model absent: ${model}"; return 0
    fi

    log "arm=${arm} START model=${model} keep=${keep} (no heal)"
    local pids=()
    for i in $(seq 0 $((NGPU-1))); do
        CUDA_VISIBLE_DEVICES="$i" "$PY" "$SCRIPT" \
            --base_model "$model" --any_family --keep_front_layers "$keep" \
            --output_name "$arm" \
            --num_shards "$NGPU" --shard_index "$i" \
            --batch_size "$BS" --add_bos 0 --content_desc full \
            > "logs/a01_gate1_${arm}_shard${i}.log" 2>&1 &
        pids+=($!)
    done

    local fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

    local nshard
    nshard=$(ls "${out}"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
    log "  arm=${arm} shards ${nshard}/${NGPU}"
    if (( fail > 0 || nshard != NGPU )); then
        log "  arm=${arm} INCOMPLETE (fail=${fail}) -- NOT merging"; return 1
    fi

    "$PY" "$SCRIPT" --merge --output_name "$arm" \
        --num_shards "$NGPU" --base_model "$model" \
        >> "logs/a01_gate1_${arm}_merge.log" 2>&1

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
    log "  arm=${arm} DONE"
}

log "DRIVER START on $(hostname) -- A01 gate-1 damaged non-OLMo (no heal)"

# Three families, two damage levels each = 6 arms.
# keep=8 matches OLMo keep8 depth ratio; keep=12 matches the healthiest OLMo
# damaged rung. Running BOTH gives a two-point ladder per family that mirrors
# the OLMo keep{14,12,10,8} sweep enough to detect (or refute) monotone
# fragility-in-depth outside OLMo.
run_arm "gate1_dmg_llama2_7b_k8"     "../models/Llama--Llama2-7b"   8
run_arm "gate1_dmg_llama3_8b_k8"     "../models/Llama--Llama3-8b"   8
run_arm "gate1_dmg_qwen3_8b_k8"      "../models/Qwen3-8B-Base"      8
run_arm "gate1_dmg_llama2_7b_k12"    "../models/Llama--Llama2-7b"  12
run_arm "gate1_dmg_llama3_8b_k12"    "../models/Llama--Llama3-8b"  12
run_arm "gate1_dmg_qwen3_8b_k12"     "../models/Qwen3-8B-Base"     12

log "ALL DONE -- gate-1 damaged non-OLMo ladder"
