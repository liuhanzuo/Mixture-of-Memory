#!/usr/bin/env bash
# A01 gate-1 driver: the letter-vs-content MMLU interface on a THIRD model family.
#
# WHY THIS EXISTS
# ---------------
# A01's Kill condition includes: "第三家族和第二 benchmark 均不复现 interface failure".
# So this run can KILL A01's generality claim. The claim under test is that the
# *letter* MC interface is an unreliable instrument -- it decays toward a constant
# predictor and its argmax is broken by exact ties -- while the *content* interface
# (scoring the choice text, label-free) stays input-driven. So far that is only
# established on OLMo-2. If Llama-2 and Llama-3 both show a healthy letter
# interface well above the best-constant floor, A01's "instrument validity before
# comparison" framing does NOT generalize and must be narrowed to OLMo-2.
#
# WHAT MAKES THIS A VALID CONTRAST
# --------------------------------
# The ONLY thing that differs from the archived OLMo-2 runs is the model class:
# `--any_family` swaps Olmo2ForCausalLM for AutoModelForCausalLM in
# eval_olmo2_probe2_ppl.load_base_model_any_family. Tokenisation (AutoTokenizer),
# prompt construction, truncation, sharding, per-option scoring, length
# normalisation, tie detection and aggregation are the SAME code path. The OLMo
# path is untouched, so every archived number still reproduces bit-for-bit.
#
# NULLS -- do not compare to 0.25
# ------------------------------
# MMLU's construct-appropriate nulls, recomputed per model from that model's own
# per-example dump by the merge step:
#   * letter interface  -> best-constant letter (on OLMo-2 this was always-D = 0.2689)
#   * content interface -> longest-option, split-tie (on OLMo-2 this was 0.2845)
# The gold-label distribution is a property of MMLU, not of the model, so the
# letter floor is expected to land at 0.2689 again; the point of recomputing is
# that we never hardcode it.
#
# PROTOCOL (matches every archived run in this project)
#   chat_template = False   (these are BASE LMs; chat templates are unfair)
#   --add_bos 0             (no special tokens prepended)
#   no system prompt, no few-shot
#   fp32 master weights, bf16 autocast forward
#
# INTEGRITY
#   8/8 shard files asserted present AND n == 14042 asserted before any merge is
#   trusted. A silent 5-of-8 merge has destroyed a result set in this project before.
#
# USAGE
#   bash scripts/_a01_gate1_driver_21.sh
#   # idempotent: an arm whose summary.json already exists is skipped.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
NGPU="${NGPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXPECT_N=14042

cd "$ROOT"

# Offline + local dataset cache. The MMLU parquet cache is on wzc1
# (data/hf_datasets_cache/cais___mmlu, 162 MB); without these the harness tries
# to reach the Hub and dies with ConnectionError.
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Do not starve the concurrent CAST Dolmino download (PID 25999) of CPU.
# .21 has 256 cores; 8 eval procs x 4 threads leaves plenty for the downloader.
export OMP_NUM_THREADS=4

mkdir -p logs olmo2_mmlu_content_results "$HF_DATASETS_CACHE"

SCRIPT=scripts/eval_olmo2_mmlu_content.py
PROGRESS=logs/a01_gate1_progress.log

log_progress() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROGRESS"; }

# arm_name  model_path
run_arm() {
    local arm="$1" model="$2"
    local out="olmo2_mmlu_content_results/${arm}"

    if [[ -f "${out}/summary.json" ]]; then
        log_progress "SKIP arm=${arm} (summary.json exists)"
        return 0
    fi
    if [[ ! -d "$model" ]]; then
        log_progress "SKIP arm=${arm} -- model dir absent: ${model}"
        return 0
    fi

    log_progress "arm=${arm} START model=${model} ${NGPU} shards"
    local pids=()
    for i in $(seq 0 $((NGPU-1))); do
        CUDA_VISIBLE_DEVICES="$i" "$PY" "$SCRIPT" \
            --base_model "$model" --any_family \
            --output_name "$arm" \
            --num_shards "$NGPU" --shard_index "$i" \
            --batch_size "$BATCH_SIZE" --add_bos 0 \
            --content_desc full \
            > "logs/a01_gate1_${arm}_shard${i}.log" 2>&1 &
        pids+=($!)
    done
    log_progress "arm=${arm} pids=${pids[*]}"

    local fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { fail=1; log_progress "arm=${arm} pid=${pid} FAILED"; }
    done

    # Hard shard-completeness gate: never merge a partial set.
    local nshard
    nshard=$(ls "${out}"/per_example_mmlu_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
    log_progress "arm=${arm} shards present ${nshard}/${NGPU}"
    if (( fail > 0 || nshard != NGPU )); then
        log_progress "arm=${arm} INCOMPLETE (fail=${fail} shards=${nshard}/${NGPU}) -- NOT merging"
        return 1
    fi

    "$PY" "$SCRIPT" --merge --output_name "$arm" \
        --num_shards "$NGPU" --base_model "$model" \
        >> "logs/a01_gate1_${arm}_merge.log" 2>&1 || {
        log_progress "arm=${arm} MERGE command failed"; return 1; }

    # Assert the merged item count, then report the two headline accuracies.
    "$PY" - "$out" "$EXPECT_N" <<'PY' | tee -a "$PROGRESS"
import json, sys
out, expect = sys.argv[1], int(sys.argv[2])
d = json.load(open(f"{out}/summary.json"))
n = d.get("n", d.get("n_total"))
if n != expect:
    raise SystemExit(f"MERGE INTEGRITY FAIL {out}: n={n} != expected {expect}")
print(f"  MERGE OK {out}: n={n} letter={d.get('letter_acc')} "
      f"content_norm={d.get('content_norm_acc')} n_nan={d.get('n_nan')}")
PY
    log_progress "arm=${arm} DONE"
}

log_progress "DRIVER START on $(hostname) ngpu=${NGPU} bs=${BATCH_SIZE} (A01 gate-1, third model family)"

# Third family #1: Llama-2-7B. Different tokenizer (32k SP vocab), different
# pretraining corpus, different RoPE config from OLMo-2 -- a genuine family change.
run_arm "gate1_llama2_7b"  "../models/Llama--Llama2-7b"

# Third family #2: Llama-3-8B. Different again (128k BPE vocab, GQA).
run_arm "gate1_llama3_8b"  "../models/Llama--Llama3-8b"

# Fourth: Qwen3-8B-Base, to make the generality claim harder to dismiss as
# "Llama-specific". Base (not instruct) so the no-chat-template protocol is fair.
run_arm "gate1_qwen3_8b_base" "../models/Qwen3-8B-Base"

log_progress "ALL DONE"
