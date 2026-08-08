#!/usr/bin/env bash
# A01 gate-3 driver: run fp32-vs-bf16 paired McNemar on OLMo-2-7B MMLU for
# base_full + keep8@121k arms. 8-shard across 8 GPUs; merge asserts full n.
# Idempotent: skips any (arm, phase) whose expected output already exists.
#
# Usage:
#   bash scripts/_a01_gate3_driver_73.sh
#
# Emits:
#   logs/a01_gate3_progress.log        — human-readable progress
#   logs/a01_gate3_${arm}_shard${i}.log — per-shard logs
#   results/a01_gate3/dtype_runs/${arm}/    — per-shard raw
#   results/a01_gate3/dtype_runs/${arm}/summary.json  — merged result
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
NGPU="${NGPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXPECT_N=14042  # MMLU total examples

cd "$ROOT"
mkdir -p logs results/a01_gate3/dtype_runs

SCRIPT="proposal/active/A01-null-calibration-methodology/code/a01_gate3_fp32_vs_bf16.py"
BASE_MODEL="../models/OLMo-2-1124-7B"

log_progress() {
    local ts
    ts="$(date '+%H:%M:%S')"
    echo "[$ts] $*" | tee -a logs/a01_gate3_progress.log
}

run_arm() {
    local arm_name="$1"
    local extra_args="$2"
    local out_dir="results/a01_gate3/dtype_runs/${arm_name}"
    # NB: the gate-3 harness writes dtype_summary.json, NOT summary.json. Using
    # the wrong name here previously made a SUCCESSFUL merge look like a failure,
    # which aborted the driver and idled 8 GPUs for 20 minutes.
    local summary="${out_dir}/dtype_summary.json"

    if [[ -f "$summary" ]]; then
        log_progress "SKIP arm=${arm_name} (dtype_summary.json exists)"
        return 0
    fi

    log_progress "arm=${arm_name} START, ${NGPU} shards in parallel"
    local pids=()
    for i in $(seq 0 $((NGPU-1))); do
        local shard_log="logs/a01_gate3_${arm_name}_shard${i}.log"
        CUDA_VISIBLE_DEVICES="$i" "$PY" "$SCRIPT" \
            --base_model "$BASE_MODEL" \
            --output_name "$arm_name" \
            --num_shards "$NGPU" --shard_index "$i" \
            --batch_size "$BATCH_SIZE" \
            $extra_args \
            > "$shard_log" 2>&1 &
        pids+=($!)
    done
    log_progress "arm=${arm_name} launched pids=${pids[*]}"

    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=1
            log_progress "arm=${arm_name} shard pid=${pid} FAILED"
        fi
    done
    if (( fail > 0 )); then
        log_progress "arm=${arm_name} at least one shard failed; NOT merging"
        return 1
    fi
    log_progress "arm=${arm_name} all ${NGPU} shards done, merging"

    "$PY" "$SCRIPT" \
        --merge \
        --output_name "$arm_name" \
        --num_shards "$NGPU" \
        --expect_n "$EXPECT_N" \
        --base_model "$BASE_MODEL" \
        $extra_args \
        >> logs/a01_gate3_${arm_name}_merge.log 2>&1
    if [[ -f "$summary" ]]; then
        log_progress "arm=${arm_name} MERGE OK -> ${summary}"
    else
        log_progress "arm=${arm_name} MERGE FAILED"
        return 1
    fi
}

log_progress "DRIVER START on $(hostname) ngpu=${NGPU} bs=${BATCH_SIZE}"

# ARM 1: base full 32-layer (no ckpt)
run_arm "7B_base_dtype" ""

# ARM 2: keep8 heal @ step121000 (most damaged in the ladder; most ties)
run_arm "7B_keep8_step121000_dtype" \
    "--ckpt outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt --keep_front_layers 8 --n_fresh_layers 2"

# ---------------------------------------------------------------------------
# ARMS 3-4 (added 2026-08-09): the two intermediate rungs.
#
# Arms 1-2 established that the tie rate is a bf16 readout of logit compression
# (gap median 1.1185 on base vs 0.2500 on keep8) and that removing the ties
# changes nothing (McNemar p=1.000 and p=0.570). What that pair CANNOT show is
# whether the compression is monotone in damage or just an endpoint contrast.
# Adding keep14 and keep12 gives a 4-point ladder in (tie_rate, gap_median),
# which is the quantity that connects this gate to B04's per-item margin result
# (Spearman(core6, frac<0.005) = -1.00 at p=0.0028 over six rungs). If gap
# median falls monotonically base -> keep14 -> keep12 -> keep8, then gate-3 and
# B04 are measuring the same underlying compression through two different
# instruments, which is a much stronger statement than either alone.
# ---------------------------------------------------------------------------
run_arm "7B_keep14_step200000_dtype" \
    "--ckpt outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt --keep_front_layers 14 --n_fresh_layers 2"

run_arm "7B_keep12_step124000_dtype" \
    "--ckpt outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt --keep_front_layers 12 --n_fresh_layers 2"

log_progress "ALL DONE"
