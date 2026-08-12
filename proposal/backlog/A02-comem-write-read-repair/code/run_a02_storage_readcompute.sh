#!/usr/bin/env bash
# ============================================================================
# A02 reframe gate launcher — storage / read-compute for high-reuse workloads.
#
# Runs the (store_L x tier x proc) grid for
# `bench_a02_storage_readcompute.py`, then aggregates to N* + storage ratios.
#
# WHY THE GRID LOOKS LIKE THIS
#   store_L in {32k, 128k, 1M} and read_sample_length=32k are chosen to match
#   the RELEASED P1.8 artifact (paperA/artifacts/p1_8_serving) exactly, so the
#   `comem` and `j0_top12` columns are directly comparable to published numbers
#   and act as a cross-check on this driver. The NEW column is `c1_all` -- the
#   arm phase-1 actually used as C1 (pack-all, full depth, per query), which the
#   published artifact never costed.
#
# NODE ETIQUETTE (.82 hosts an A03 eval watcher)
#   The A03 watcher wakes ~every 2.8h and wants all 8 GPUs for ~4 min. It
#   self-protects: it REFUSES to start while >8000 MiB is held and retries after
#   60 s, so it can never OOM against this bench -- it only waits. To keep that
#   wait short this launcher uses a BOUNDED GPU pool (default 4 of 8 GPUs,
#   leaving 4 idle) and each unit is a self-contained single-GPU process that
#   writes its own JSON, so a transient failure costs ONE retryable cell.
#
# Usage (on .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash proposal/backlog/A02-comem-write-read-repair/code/run_a02_storage_readcompute.sh \
#     >logs/a02_storage_readcompute.out 2>&1 &
#   # omit RUN=1 for a dry preview (prints commands, no GPU work).
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
RUN="${RUN:-0}"
GPUS_STR="${GPUS:-0 1 2 3}"          # bounded pool: leave GPUs free for A03

DRIVER="proposal/backlog/A02-comem-write-read-repair/code/bench_a02_storage_readcompute.py"
MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUTDIR="${OUTDIR:-bench_results/a02_storage_readcompute}"

STORE_LENGTHS="${STORE_LENGTHS:-32k 128k 1M}"
READ_SAMPLE_LENGTH="${READ_SAMPLE_LENGTH:-32k}"
GEN_LENGTHS="${GEN_LENGTHS:-1 32 128}"
QUERY_COUNTS="${QUERY_COUNTS:-1 4 16 64 256 1024}"
TIERS="${TIERS:-gpu cpu}"
N_PROCS="${N_PROCS:-3}"
N_REPEAT="${N_REPEAT:-5}"
WARMUP="${WARMUP:-2}"
TASK="${TASK:-niah_multikey_1}"

COMMON="--model_path $MODEL --lora_adapter $LORA --resume_j 12 \
--topk 12 --iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
--seed 42 --read_sample_length $READ_SAMPLE_LENGTH --task $TASK \
--n_repeat $N_REPEAT --warmup $WARMUP --output_dir $OUTDIR"

LOGDIR="logs/a02_storage_readcompute"
DONEDIR="$LOGDIR/done"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

echo "============================================================"
echo "[a02] PROJECT_ROOT=$PROJECT_ROOT  PYBIN=$PYBIN  RUN=$RUN"
echo "[a02] store_L=[$STORE_LENGTHS] tiers=[$TIERS] G=[$GEN_LENGTHS] N=[$QUERY_COUNTS]"
echo "[a02] procs/cell=$N_PROCS n_repeat=$N_REPEAT warmup=$WARMUP read_pack=$READ_SAMPLE_LENGTH"
echo "[a02] GPU pool=[$GPUS_STR] (bounded on purpose: A03 watcher shares this node)"
[ "$RUN" != "1" ] && echo "[a02] *** DRY-RUN: printing commands only ***"
echo "============================================================"

if [ "$RUN" = "1" ]; then
    cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }
    export PYTHONUNBUFFERED=1
    export PYTHONHASHSEED=0          # deterministic hash((task,length)) sample seed
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$PROJECT_ROOT/.hf_cache"
    unset http_proxy https_proxy all_proxy
    mkdir -p "$LOGDIR" "$DONEDIR" "$OUTDIR"
fi

# ---- STEP 0: LoRA-sha manifest gate ----------------------------------------
echo "[a02] STEP 0: manifest / LoRA-sha gate"
if [ "$RUN" = "1" ]; then
    CUDA_VISIBLE_DEVICES=$(echo $GPUS_STR | awk '{print $1}') \
        $PYBIN $DRIVER --mode manifest $COMMON >"$LOGDIR/manifest.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "[a02] MANIFEST ABORTED"; tail -20 "$LOGDIR/manifest.out"; exit 3
    fi
    grep -o 'match=[A-Za-z]*' "$LOGDIR/manifest.out" || true
else
    echo "  \$ CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON"
fi

# ---- STEP 1: build the unit queue -----------------------------------------
echo "[a02] STEP 1: unit grid (store_L x tier x proc)"
NJOBS=0
[ "$RUN" = "1" ] && : > "$QUEUE"
for sl in $STORE_LENGTHS; do
  for tier in $TIERS; do
    for p in $(seq 0 $((N_PROCS - 1))); do
      echo "    unit: $sl|$tier|$p"
      [ "$RUN" = "1" ] && echo "$sl|$tier|$p" >> "$QUEUE"
      NJOBS=$((NJOBS + 1))
    done
  done
done
echo "[a02] queued $NJOBS units (each single-GPU, self-contained, retryable)"

if [ "$RUN" != "1" ]; then
    echo "[a02] example unit (proc0 also runs the store==recompute + on-disk probes):"
    echo "  \$ CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode serve $COMMON \\"
    echo "      --store_length 32k --tier gpu --proc_id 0 --verify --measure_ondisk \\"
    echo "      --gen_lengths $GEN_LENGTHS"
    echo "[a02] aggregate (CPU) after all units:"
    echo "  \$ $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR --query_counts $QUERY_COUNTS --expect_procs $N_PROCS"
    exit 0
fi

# ---- STEP 2: bounded flock task pool --------------------------------------
read -r -a GPUS <<< "$GPUS_STR"
pop_job() {
    ( flock -x 200
      local first; first=$(head -n 1 "$QUEUE")
      if [ -n "$first" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
      echo "$first" ) 200>"$LOCK"
}
worker() {
    local gpu="$1"
    while true; do
        local job; job=$(pop_job); [ -z "$job" ] && break
        IFS='|' read -r sl tier p <<< "$job"
        local tag="serve_${sl}_${tier}_p${p}"
        [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
        # proc0 of each cell carries the extra integrity + real-file-size probes
        local extra=""
        [ "$p" = "0" ] && extra="--verify --measure_ondisk"
        echo "[gpu$gpu] START $tag $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode serve $COMMON \
            --store_length "$sl" --tier "$tier" --proc_id "$p" $extra \
            --gen_lengths $GEN_LENGTHS \
            >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
        echo "[gpu$gpu] END $tag rc=$? $(date +%H:%M:%S)"
    done
    echo "[gpu$gpu] worker exit"
}
echo "[a02] STEP 2: launching ${#GPUS[@]} workers"
pids=()
for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
for pp in "${pids[@]}"; do wait "$pp"; done
echo "[a02] all workers done."

# ---- STEP 3: aggregate -----------------------------------------------------
echo "[a02] STEP 3: aggregate -> N* + storage ratios"
$PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" \
    --query_counts $QUERY_COUNTS --expect_procs $N_PROCS \
    >"$LOGDIR/aggregate.out" 2>&1
rc=$?
cat "$LOGDIR/aggregate.out"
if [ $rc -ne 0 ]; then
    echo "[a02] AGGREGATE FAILED rc=$rc (likely an incomplete cell -- see above)."
    echo "[a02] Re-run the missing units; do NOT merge partial cells silently."
    exit 5
fi
echo "[a02] COMPLETE -> $OUTDIR/a02_storage_readcompute_aggregate.json"
