#!/usr/bin/env bash
# ============================================================================
# P0.13 — same-pack / same-LoRA / same-examples quality<->latency closed loop
# on node .82 (28.82.250.82:36000, 8xH20, diskB). FINAL required model run.
#
# Two arms differ ONLY in resume_j:
#   Arm A: resume_j=0  + flagship rank-32 LoRA, replay all 36 layers.
#   Arm B: resume_j=12 + SAME LoRA, replay upper 24 layers from cached h12.
# For each example the pack is built ONCE (forward-free iter_bm25 selection) and
# BOTH arms run on that identical pack => strict 1:1 pairing.
#
# FLAGSHIP RULER QCMem config (mandated, identical to the headline row):
#   selector=iter_bm25  topk=12  iter_hop_topk=4  sink=bos  chunk_size=512
#   chat_template=False  enable_thinking=False  bf16  sdpa
#
# Cohort B = {niah_single_3, niah_multikey_1, variable_tracking}
#          x {8k,16k,32k,64k,128k}  n=100  => 15 cells.
# Sharding: each cell split into 4 shards x n=25 (i % 4). 15 cells x 4 = 60 jobs
# across an 8-GPU flock task pool (dynamic load balance; 128k jobs never stall).
#
# Env: /opt/conda/envs/torch-base/bin/python (py3.14 / torch2.13 / tf5.5.4 /
#      peft0.19.1) — the SAME env P0.12 acceptance used (.venv is broken on .82).
#
# Usage on .82:
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   MODE=smoke|full  setsid nohup bash scripts/launch_p0_13_82.sh >logs/p0_13.out 2>&1 &
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }

PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODE="${MODE:-full}"                 # smoke | full
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0               # deterministic hash((task,length)) sample seed
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export http_proxy="" https_proxy="" all_proxy=""

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUTDIR="${OUTDIR:-bench_results/p0_13_quality_latency}"
DRIVER="scripts/bench_p0_13_quality_latency.py"
COMMON="--model_path $MODEL --lora_adapter $LORA --resume_j_a 0 --resume_j_b 12 \
--topk 12 --iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
--seed 42 --output_dir $OUTDIR"

GPUS=(0 1 2 3 4 5 6 7)
LOGDIR="logs/p0_13"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR" "$OUTDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

# ---- STEP 0: manifest (strict-fix gate). Aborts (exit 3) on hash mismatch. ----
echo "[p0.13] STEP 0: manifest / strict-fix gate on GPU 0"
CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON \
    >"$LOGDIR/manifest.out" 2>&1
if [ $? -ne 0 ]; then
    echo "[p0.13] MANIFEST ABORTED — see $LOGDIR/manifest.out; refusing to run."
    cat "$LOGDIR/manifest.out" | tail -20
    exit 3
fi
echo "[p0.13] manifest OK."

# ---- STEP 1: 1-cell n=4 smoke on GPU 0 (confirm paired output + scores) ----
if [ "$MODE" = "smoke" ] || [ "${DO_SMOKE:-1}" = "1" ]; then
    echo "[p0.13] STEP 1: smoke niah_single_3/8k n=4 on GPU 0"
    CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode quality $COMMON \
        --task niah_single_3 --length 8k --limit 4 --num_shards 1 --shard_index 0 \
        --output_dir "$OUTDIR/_smoke" >"$LOGDIR/smoke.out" 2>&1
    rc=$?
    echo "[p0.13] smoke rc=$rc; tail:"; tail -8 "$LOGDIR/smoke.out"
    if [ $rc -ne 0 ]; then echo "[p0.13] SMOKE FAILED — aborting."; exit 4; fi
    if [ "$MODE" = "smoke" ]; then echo "[p0.13] smoke-only mode done."; exit 0; fi
fi

# ---- STEP 2: build the quality job queue (15 cells x 4 shards = 60 jobs) ----
TASKS=(niah_single_3 niah_multikey_1 variable_tracking)
LENGTHS=(128k 64k 32k 16k 8k)   # LPT: longest first so they never stall the pool
: > "$QUEUE"
for len in "${LENGTHS[@]}"; do
    for task in "${TASKS[@]}"; do
        for sh in 0 1 2 3; do
            echo "quality|$task|$len|$sh|4" >> "$QUEUE"
        done
    done
done
# latency jobs: 3 independent procs on a fixed representative pack (16k niah_single_3)
for p in 0 1 2; do
    echo "latency|niah_single_3|16k|$p|3" >> "$QUEUE"
done
echo "[p0.13] queued $(wc -l < "$QUEUE") jobs (60 quality + 3 latency)."

# ---- STEP 3: 8-GPU flock task pool ----
pop_job() {
    (
        flock -x 200
        local first
        first=$(head -n 1 "$QUEUE")
        if [ -n "$first" ]; then
            tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
        fi
        echo "$first"
    ) 200>"$LOCK"
}

worker() {
    local gpu="$1"
    while true; do
        local job; job=$(pop_job)
        [ -z "$job" ] && break
        IFS='|' read -r kind task len a b <<< "$job"
        local tag="${kind}_${task}_${len}_${a}"
        local done_marker="$DONEDIR/$tag.done"
        [ -f "$done_marker" ] && { echo "[gpu$gpu] SKIP $tag (done)"; continue; }
        echo "[gpu$gpu] START $job"
        if [ "$kind" = "quality" ]; then
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode quality $COMMON \
                --task "$task" --length "$len" --limit 100 \
                --num_shards "$b" --shard_index "$a" \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$done_marker"
        else
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode latency $COMMON \
                --task "$task" --length "$len" --proc_id "$a" \
                --example_index 0 --warmup 3 --n_repeat 20 \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$done_marker"
        fi
        echo "[gpu$gpu] END   $job rc=$?"
    done
    echo "[gpu$gpu] worker exit"
}

echo "[p0.13] STEP 3: launching 8 GPU workers"
pids=()
for g in "${GPUS[@]}"; do
    worker "$g" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[p0.13] all workers done."

# ---- STEP 4: aggregate ----
echo "[p0.13] STEP 4: aggregate"
$PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" >"$LOGDIR/aggregate.out" 2>&1
cat "$LOGDIR/aggregate.out"
echo "[p0.13] COMPLETE."
