#!/usr/bin/env bash
# ============================================================================
# P1.8 — repeated-query SERVING curve: CoMem (j=12+LoRA, write-once) vs matched
# j=0 BM25 raw-text replay. Zero-training latency/throughput bench (DRY-DEFAULT).
#
# Positively answers "under which workload does CoMem STRICTLY dominate a matched
# raw-text replay?" by measuring the real per-component serving cost across the
# store-size L x query-count Q x generation-length G matrix, on two store tiers
# (gpu-resident / cpu-pinned), then emitting the (Q x G) crossover grid.
#
# TWO MAIN ARMS (matched: SAME example, SAME iter_bm25 top-12 pack, SAME Read/decode
# primitives imported bit-identically from eval_p016_e0_write_control -> bench_p1_7
# -> bench_p0_13):
#   comem : CoMem j=12+LoRA. One-time O(L) h12 Write over N=L/512 chunks -> persistent
#           store (8192 B/tok). Per query: select + fetch(top-12 pack H2D) + 24-layer
#           Read + G decode. Big one-time cost, cheap per query.
#   j0    : BM25 raw-text replay j=0. One-time = BM25 index (cheap, ~4 B/tok store).
#           Per query: select + fetch(token ids) + FULL 36-layer replay + G decode.
#           ~no one-time cost, expensive per query.
#   (optional --with_fullctx reference arm: full-context prefill+decode over all L
#    tokens; OOM recorded. Reference ONLY — the paper judgement is comem vs j0.)
#
# Crossover: cumulative(Q,G) = write_once + Q*(select+fetch+read+decode(G)). Break-even
# Q*(G,L) where comem cumulative == j0 cumulative (select cancels — equal both arms).
# P0.2 analytic predicts ~Q>=17-20 @128k; larger G shifts the crossover EARLIER.
#
# FLAGSHIP config (mandated, identical to P0.13/P1.7/P0.16 headline):
#   model=models/Qwen3-8b-local  LoRA=outputs/qcmem_distill_qwen_j12_r32_4k/final
#   selector=iter_bm25  topk=12  iter_hop_topk=4  chunk_size=512  sink=bos
#   resume_j=12  chat_template=False  add_bos=0  bf16  sdpa  seed=42
#
# 1 GPU is SUFFICIENT (pure latency bench). Each (L, tier, proc) is a self-contained
# single-GPU unit; the flock pool below just fans units across GPUs for wall-clock.
# >=3 independent procs per (L,tier) cell => independent repeats (median + tails).
#
# ★ DRY-BY-DEFAULT: prints the manifest / selfcheck / serve / aggregate commands but
#   runs NO GPU forward. MAIN launches the real bench by exporting RUN=1 on a FREE
#   diskB H20 node (never self-runs, never contends with a live training job).
#
# Usage (MAIN, on a FREE diskB H20 node — e.g. .104):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_p1_8_serving.sh \
#     >logs/p1_8_serving.out 2>&1 &
#   # dry preview (default): omit RUN=1 — prints commands only.
#
# Single-GPU manual (one cell):
#   CUDA_VISIBLE_DEVICES=0 <PYBIN> scripts/bench_p1_8_serving_curve.py --mode serve \
#     --store_length 128k --tier gpu --proc_id 0
# ============================================================================
set -uo pipefail

# ---- overridable env (defaults tuned for diskB; NO auto-run) ----------------
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYBIN="${PYTHON_BIN:-.venv/bin/python}"   # diskB H20 -> /opt/conda/envs/torch-base/bin/python
RUN="${RUN:-0}"                            # 0 = DRY (print only) ; 1 = execute
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"        # GPU pool for the flock scheduler

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUTDIR="${OUTDIR:-bench_results/p1_8_serving}"
DRIVER="scripts/bench_p1_8_serving_curve.py"

# matrix axes (store L; query counts Q; generation lengths G; tiers; procs)
STORE_LENGTHS="${STORE_LENGTHS:-32k 128k 1M}"
QUERY_COUNTS="${QUERY_COUNTS:-1 4 16 32 64}"
GEN_LENGTHS="${GEN_LENGTHS:-1 32 128 512}"
TIERS="${TIERS:-gpu cpu}"
N_PROCS="${N_PROCS:-3}"                     # >=3 independent processes per (L,tier)
N_REPEAT="${N_REPEAT:-5}"
WARMUP="${WARMUP:-2}"
READ_SAMPLE_LENGTH="${READ_SAMPLE_LENGTH:-32k}"  # L-INDEPENDENT Read pack source
TASK="${TASK:-niah_multikey_1}"
WITH_FULLCTX="${WITH_FULLCTX:-0}"           # 1 => also time full-ctx reference (OOM-safe)

FULLCTX_FLAG=""; [ "$WITH_FULLCTX" = "1" ] && FULLCTX_FLAG="--with_fullctx"
COMMON="--model_path $MODEL --lora_adapter $LORA --resume_j 12 \
--topk 12 --iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
--seed 42 --read_sample_length $READ_SAMPLE_LENGTH --task $TASK \
--n_repeat $N_REPEAT --warmup $WARMUP --output_dir $OUTDIR $FULLCTX_FLAG"

echo "============================================================"
echo "[p1.8] PROJECT_ROOT=$PROJECT_ROOT"
echo "[p1.8] PYBIN=$PYBIN  RUN=$RUN"
echo "[p1.8] STORE_L=[$STORE_LENGTHS] Q=[$QUERY_COUNTS] G=[$GEN_LENGTHS] tiers=[$TIERS]"
echo "[p1.8] procs/cell=$N_PROCS n_repeat=$N_REPEAT warmup=$WARMUP read_pack=$READ_SAMPLE_LENGTH"
echo "[p1.8] MODEL=$MODEL LORA=$LORA OUTDIR=$OUTDIR fullctx=$WITH_FULLCTX"
[ "$RUN" != "1" ] && echo "[p1.8] *** DRY-RUN (RUN!=1): printing commands only, no forward ***"
echo "============================================================"

LOGDIR="logs/p1_8_serving"
DONEDIR="$LOGDIR/done"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

_do() {
    echo "  \$ $*"
    if [ "$RUN" = "1" ]; then eval "$@"; return $?; fi
    return 0
}

if [ "$RUN" = "1" ]; then
    cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }
    export PYTHONUNBUFFERED=1
    export PYTHONHASHSEED=0             # deterministic hash((task,length)) sample seed
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$PROJECT_ROOT/.hf_cache"
    export http_proxy="" https_proxy="" all_proxy=""
    mkdir -p "$LOGDIR" "$DONEDIR" "$OUTDIR"
fi

# ---- STEP 0: manifest gate (LoRA sha == flagship). Aborts on mismatch. ----
echo "[p1.8] STEP 0: manifest / LoRA-sha gate on GPU 0"
_do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON >\"$LOGDIR/manifest.out\" 2>&1"
if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
    echo "[p1.8] MANIFEST ABORTED — see $LOGDIR/manifest.out"; tail -20 "$LOGDIR/manifest.out"; exit 3
fi

# ---- STEP 1: selfcheck gate (store-fetched h12 == fresh recompute, bit-identical) ----
echo "[p1.8] STEP 1: selfcheck (store==recompute + finite + pack-pairing) on GPU 0"
_do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode selfcheck $COMMON \
--store_length 32k --tier gpu --proc_id 0 >\"$LOGDIR/selfcheck.out\" 2>&1"
if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
    echo "[p1.8] SELFCHECK FAILED — cache reuse changed the Read inputs; aborting."
    tail -30 "$LOGDIR/selfcheck.out"; exit 4
fi

# ---- STEP 2: build the serve job queue (L x tier x proc units) ----
echo "[p1.8] STEP 2: serve job grid (store_L x tier x proc)"
NJOBS=0
if [ "$RUN" = "1" ]; then : > "$QUEUE"; fi
for sl in $STORE_LENGTHS; do
    for tier in $TIERS; do
        for p in $(seq 0 $((N_PROCS - 1))); do
            echo "    job: serve|$sl|$tier|$p"
            [ "$RUN" = "1" ] && echo "serve|$sl|$tier|$p" >> "$QUEUE"
            NJOBS=$((NJOBS + 1))
        done
    done
done
echo "[p1.8] queued $NJOBS serve jobs (each single-GPU, self-contained)."

# per-unit command template (dry preview of what each worker runs):
FIRST_L=$(echo $STORE_LENGTHS | awk '{print $1}'); FIRST_T=$(echo $TIERS | awk '{print $1}')
echo "[p1.8] example per-unit command (proc 0 also runs --verify store==recompute gate):"
echo "    CUDA_VISIBLE_DEVICES=<gpu> $PYBIN $DRIVER --mode serve $COMMON \\"
echo "        --store_length $FIRST_L --tier $FIRST_T --proc_id 0 --verify \\"
echo "        --gen_lengths $GEN_LENGTHS"

# ---- STEP 3: 8-GPU flock task pool (only when RUN=1) ----
if [ "$RUN" = "1" ]; then
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
            IFS='|' read -r kind sl tier p <<< "$job"
            local tag="${kind}_${sl}_${tier}_p${p}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            local verify=""; [ "$p" = "0" ] && verify="--verify"
            echo "[gpu$gpu] START $job"
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode serve $COMMON \
                --store_length "$sl" --tier "$tier" --proc_id "$p" $verify \
                --gen_lengths $GEN_LENGTHS \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[p1.8] STEP 3: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for pp in "${pids[@]}"; do wait "$pp"; done
    echo "[p1.8] all workers done."

    # ---- STEP 4: aggregate -> (Q x G) crossover grid + P0.2 cross-check ----
    echo "[p1.8] STEP 4: aggregate (Q x G crossover)"
    $PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" \
        --query_counts $QUERY_COUNTS >"$LOGDIR/aggregate.out" 2>&1
    cat "$LOGDIR/aggregate.out"
    echo "[p1.8] COMPLETE. crossover json: $OUTDIR/p1_8_serving_aggregate.json"
else
    echo "[p1.8] STEP 3/4 (workers + aggregate) — DRY: not launched."
    echo "[p1.8] aggregate (CPU) command MAIN runs after all units finish:"
    echo "    $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR --query_counts $QUERY_COUNTS"
    echo "[p1.8] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
