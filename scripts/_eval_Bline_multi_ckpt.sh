#!/usr/bin/env bash
# ============================================================================
# B-line multi-checkpoint eval: BABILong qa1 + RULER + LongEval
# Checkpoints: step500, step1000, step1400
# Runs each checkpoint's full suite serially (one ckpt at a time).
# ============================================================================
set -uo pipefail

RD="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$RD"
PYBIN="${PYTHON_BIN:-$RD/.venv/bin/python}"

export WANDB_MODE=offline
export HF_HOME="$RD/.hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RD:$RD/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"

CKPT_DIR="outputs/mem_space_B_inwindow_b200_ckpt"
ACFG="$CKPT_DIR/adapter_config.json"
MODEL="models/Meta-Llama-3-8B"
LOGDIR="logs/eval_Bline_multi_ckpt"
mkdir -p "$LOGDIR"

# Checkpoints to eval
STEPS=(500 1000 1400)

# ---- wait_for_ckpt: poll until file exists and size is stable (>1GB) ----
wait_for_ckpt() {
    local ckpt="$1"
    local name="$2"
    echo "[$(date)] Waiting for $name ($ckpt)..."
    while true; do
        if [ ! -f "$ckpt" ]; then
            sleep 60; continue
        fi
        local sz1; sz1=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
        sleep 15
        local sz2; sz2=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
        if [ "$sz1" -eq "$sz2" ] && [ "$sz1" -gt "20000000000" ]; then
            echo "[$(date)] $name ready: $ckpt ($(numfmt --to=iec "$sz1"))"
            return 0
        fi
        echo "[$(date)] $name still transferring: $(numfmt --to=iec "$sz2")/~23G..."
        sleep 60
    done
}

# ---- run_babilong: BABILong qa1 4k/8k/16k/32k via taskpool 2group ----
run_babilong() {
    local step="$1"
    local ckpt_name="Bline_step${step}"
    local ckpt_file="$CKPT_DIR/full_model_step$(printf '%06d' "$step").pt"

    echo "[$(date)] === BABILong eval: $ckpt_name ==="
    RUN_PREFIX="$ckpt_name" \
    CKPT_FILES="$ckpt_file" \
    CK_NAMES="$ckpt_name" \
    ADAPTER_CONFIG="$ACFG" \
    MODEL="$MODEL" \
    TASKS="qa1" \
    LENGTHS="4k 8k 16k 32k" \
    CHUNK_SIZE=1024 \
    LIMIT=100 \
    NSHARD=4 \
    EXTRA_ARGS="--swa_eval_chunks 0" \
    NUM_GROUPS=2 \
    bash scripts/_eval_taskpool_2group.sh \
        >> "$LOGDIR/babilong_step${step}.log" 2>&1
    local rc=$?
    echo "[$(date)] BABILong $ckpt_name done. exit=$rc"
    echo "--- BABILong scores for step${step} ---"
    $PYBIN scripts/score_nested_babilong.py "babilong_results/$ckpt_name" 2>&1 | tail -20
}

# ---- run_ruler: RULER 4 tasks x 4 lengths across 8 GPUs ----
run_ruler() {
    local step="$1"
    local ckpt_file="$CKPT_DIR/full_model_step$(printf '%06d' "$step").pt"
    local out_name="ruler_Bline_step${step}"
    local NS=50

    echo "[$(date)] === RULER eval: step${step} (out=$out_name) ==="
    mkdir -p ruler_results

    # Build a pool file
    local POOL; POOL=$(mktemp)
    local LOCK; LOCK=$(mktemp)
    for t in niah_single_1 niah_single_2 niah_multikey_1 variable_tracking; do
        for L in 4k 8k 16k 32k; do
            echo "$t $L" >> "$POOL"
        done
    done

    # Worker runs one cell at a time from the pool
    for gpu in 0 1 2 3 4 5 6 7; do
        (
            while true; do
                local cell=""
                exec 9>"$LOCK"; flock 9
                cell=$(head -n1 "$POOL" 2>/dev/null || true)
                if [ -n "$cell" ]; then sed -i '1d' "$POOL"; fi
                flock -u 9; exec 9>&-
                [ -z "$cell" ] && break
                local tt=${cell% *}; local LL=${cell#* }
                CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_ruler_mem_space.py \
                    --model_type mem_space \
                    --model_path "$MODEL" --checkpoint "$ckpt_file" \
                    --adapter_config "$ACFG" \
                    --output_name "$out_name" \
                    --chunk_size 1024 --swa_eval_chunks 0 \
                    --tasks "$tt" --lengths "$LL" --num_samples $NS \
                    >> "$LOGDIR/ruler_step${step}_gpu${gpu}.log" 2>&1
            done
        ) &
    done
    wait
    rm -f "$POOL" "$LOCK"
    echo "[$(date)] RULER step${step} done."

    # Print summary
    echo "--- RULER scores for step${step} ---"
    local summary_file="ruler_results/${out_name}/_summary.json"
    if [ -f "$summary_file" ]; then
        $PYBIN -c "
import json, sys
d = json.load(open('$summary_file'))
for task, lengths in d.items():
    row = '  '.join(f'{L}={v[\"score\"]:.1f}' for L, v in sorted(lengths.items()))
    print(f'  {task:>25}: {row}')
"
    else
        echo "  No summary file found: $summary_file"
        for t in niah_single_1 niah_single_2 niah_multikey_1 variable_tracking; do
            for L in 4k 8k 16k 32k; do
                local rf="ruler_results/${out_name}/${t}_${L}.json"
                if [ -f "$rf" ]; then
                    $PYBIN -c "
import json; d=json.load(open('$rf'))
s=d.get('summary',{}).get('score',d.get('score','?'))
print(f'  $t $L: {s}')
" 2>/dev/null || echo "  $t $L: parse error"
                else
                    echo "  $t $L: missing"
                fi
            done
        done
    fi
}

# ---- run_longeval: LongEval 4k/8k/16k/32k, 4 GPUs in parallel ----
run_longeval() {
    local step="$1"
    local ckpt_file="$CKPT_DIR/full_model_step$(printf '%06d' "$step").pt"
    local out_name="Bline_step${step}"

    echo "[$(date)] === LongEval: step${step} (out=$out_name) ==="
    local lengths=(4k 8k 16k 32k)
    local gpus=(0 1 2 3)
    local pids=()

    for i in "${!lengths[@]}"; do
        local L="${lengths[$i]}"
        local gpu="${gpus[$i]}"
        CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_longeval_mem_space.py \
            --model_path "$MODEL" --checkpoint "$ckpt_file" \
            --adapter_config "$ACFG" \
            --output_name "$out_name" \
            --lengths "$L" --num_samples 50 \
            --chunk_size 1024 --swa_eval_chunks 0 \
            >> "$LOGDIR/longeval_step${step}_${L}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[$(date)] LongEval step${step} done."

    # Print scores
    echo "--- LongEval scores for step${step} ---"
    for L in 4k 8k 16k 32k; do
        local f="longeval_results/$out_name/longeval_${L}.json"
        if [ -f "$f" ]; then
            $PYBIN -c "
import json; d=json.load(open('$f'))
s=d['summary']; print(f'  $L: acc={s[\"accuracy\"]:.3f}  ({s[\"correct\"]}/{s[\"total\"]})')
" 2>/dev/null || echo "  $L: parse error"
        else
            echo "  $L: missing"
        fi
    done
}

# ============================================================
# Main: for each step, wait for ckpt then run all evals
# ============================================================
for step in "${STEPS[@]}"; do
    ckpt_file="$CKPT_DIR/full_model_step$(printf '%06d' "$step").pt"
    wait_for_ckpt "$ckpt_file" "Bline_step${step}"

    echo ""
    echo "=========================================="
    echo "  Starting full eval for step${step}"
    echo "=========================================="
    run_babilong "$step"
    run_ruler "$step"
    run_longeval "$step"
    echo ""
    echo "=========================================="
    echo "  step${step} ALL DONE"
    echo "=========================================="
    echo ""
done

echo "[$(date)] ALL CHECKPOINTS EVALUATED."
