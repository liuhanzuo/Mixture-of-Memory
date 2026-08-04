#!/usr/bin/env bash
# ============================================================================
# P0.20 Phase B — DENSE equal-latency frontier: dense-RAG vs CoMem.
# 8-GPU flock task-pool scheduler (DRY-BY-DEFAULT). Parallels _run_p0_20_8gpu.sh.
#
# Phase B == Phase A with ONE change: the raw-text-RAG arm's selector is swapped
# from lexical iter_bm25 to a FROZEN DENSE retriever (BGE-large-en-v1.5, CLS+L2+
# cosine — the SAME retriever P1.9 uses). The CoMem arm is UNCHANGED from Phase A
# (resume_j=12, flagship LoRA, iter_bm25 selector, pre-stored h12 fetch+H2D), so
# its TTFT anchor == Phase A's. Reader byte-identical to config#2.
#
# Research question (paperA §P0.20 Phase B): at CoMem(k=12)'s FIXED latency budget,
# how many chunks can a deployment-realistic DENSE RAG read (k_dense*), and does the
# dense retriever's better recall let latency-matched dense-RAG close/overturn the
# Phase A verdict where BM25-RAG lost?
#
# Dense selection latency is charged under TWO honest models (both reported):
#   deployment  (PRIMARY)     = passages pre-encoded/indexed OFFLINE (== CoMem
#                               pre-stores h12); online = query-encode + flat cosine
#                               search. k_dense* is frozen on THIS.
#   cold-index  (SENSITIVITY) = encode ALL passages + query + search (== P1.9's
#                               retrieval_latency_ms). Reported; k_dense* here ~0.
#
# FLAGSHIP config (mandated, identical to Phase A / P0.13 / config#2):
#   comem_selector=iter_bm25  dense_selector=dense_bge  iter_hop_topk=4  sink=bos
#   chunk_size=512  chat_template=False  enable_thinking=False  add_bos=0  bf16  sdpa
#   LoRA=outputs/qcmem_distill_qwen_j12_r32_4k/final (sha dd09cd17…, layers 12..35)
#   BGE=models/bge-large-en-v1.5 (sha 45e19549…, rev d4aa6901…, CLS+L2+cosine)
#
# Pipeline (RUN=1 executes; RUN=0/unset prints only):
#   STEP 0 manifest      — Phase A strict-fix gate (backbone+LoRA sha, layers, 168
#                          mods) PLUS BGE weight sha256 gate. Aborts (exit 3).
#   STEP 1 sanity        — LoRA disable_adapter toggle; dense determinism; P1.9
#                          reproduction cross-check (babilong/longeval/locomo where
#                          the seed convention matches); read_len arm-equality;
#                          calib/quality split disjoint. Aborts (exit 4).
#   STEP 2 CALIB LATENCY — per k in KS, on RESERVED calib docs, N_REPEAT reps x
#                          N_PROCS procs: TTFT breakdown for CoMem (unchanged) and
#                          dense-RAG (deploy + cold-index selection). k_dense* FROZEN
#                          on latency ONLY.
#   STEP 3 QUALITY pool  — 8-GPU flock task-pool over (bench,task,length,k) x shards.
#                          Both arms per example. Primary: BABILong qa1/qa2, LongEval,
#                          LoCoMo; secondary: RULER multikey.
#   STEP 4 aggregate     — freeze k_dense* (deploy PRIMARY + cold-index), frontier,
#                          paired bootstrap 95% CI + exact McNemar at anchor, verdict.
#                          ALL cells reported (never dense-wins-only).
#
# ★ DRY-BY-DEFAULT: prints every command but runs NO forward unless RUN=1.
#
# Usage (MAIN, on FREE diskB .104):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_p0_20_phaseB_dense.sh \
#     >logs/p0_20_phaseB.out 2>&1 &
#   # dry preview (default): omit RUN=1 — prints commands only.
# ============================================================================
set -uo pipefail

# ---- overridable env (defaults tuned for diskB; NO auto-run) ----------------
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYBIN="${PYTHON_BIN:-.venv/bin/python}"      # diskB .104 -> torch-base ; wzc1 -> .venv
RUN="${RUN:-0}"                              # 0 = DRY (print only) ; 1 = execute
NUM_SHARDS="${NUM_SHARDS:-4}"
LIMIT="${LIMIT:-100}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"
DO_SANITY="${DO_SANITY:-1}"

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
RETRIEVER="${RETRIEVER:-models/bge-large-en-v1.5}"
OUTDIR="${OUTDIR:-bench_results/p0_20_phaseB_dense}"
P19_OUTDIR="${P19_OUTDIR:-bench_results/p1_9_dense_rag}"
DRIVER="scripts/eval_p0_20_phaseB_dense.py"
# CoMem-arm selector. Default iter_bm25 == flagship Phase B (cross-selector control:
# dense-RAG vs bm25-CoMem). Set COMEM_SELECTOR=dense_bge for the A-P1.1 SAME-selector
# control: BOTH arms use the frozen BGE retriever, isolating the reader variable
# (full-recompute RAG j0 vs cached-h12 CoMem j12) at matched selection cost.
COMEM_SELECTOR="${COMEM_SELECTOR:-iter_bm25}"

# ---- k sweep (BOTH arms) + calibration latency reps -------------------------
KS_STR="${KS:-2 4 6 8 10 12 14 16 20 24}"
N_PROCS="${N_PROCS:-3}"
WARMUP="${WARMUP:-5}"
N_REPEAT="${N_REPEAT:-20}"
CALIB_OFFSET="${CALIB_OFFSET:-900}"
CALIB_LENGTH="${CALIB_LENGTH:-32k}"
TOL="${TOL:-0.05}"
N_BOOT="${N_BOOT:-10000}"

# ---- quality grid: primary=BABILong qa1/qa2 + LongEval + LoCoMo; secondary=RULER
QCELLS_DEFAULT="babilong|qa1|4k babilong|qa1|16k babilong|qa2|4k babilong|qa2|16k longeval|longeval|8k longeval|longeval|16k locomo|locomo|na ruler|niah_multikey_1|8k ruler|niah_multikey_1|16k"
QCELLS="${QCELLS:-$QCELLS_DEFAULT}"
QKS_STR="${QKS:-$KS_STR}"

COMMON="--model_path $MODEL --lora_adapter $LORA --retriever_path $RETRIEVER \
--iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
--comem_selector $COMEM_SELECTOR \
--seed 42 --output_dir $OUTDIR --p19_output_dir $P19_OUTDIR --limit $LIMIT \
--calib_offset $CALIB_OFFSET"

echo "============================================================"
echo "[p0.20B] PROJECT_ROOT=$PROJECT_ROOT"
echo "[p0.20B] PYBIN=$PYBIN  RUN=$RUN"
echo "[p0.20B] KS=[$KS_STR]  QKS=[$QKS_STR]  n_procs=$N_PROCS warmup=$WARMUP n_repeat=$N_REPEAT tol=$TOL"
echo "[p0.20B] QCELLS=[$QCELLS]"
echo "[p0.20B] LIMIT=$LIMIT shards=$NUM_SHARDS calib_offset=$CALIB_OFFSET calib_length=$CALIB_LENGTH"
echo "[p0.20B] MODEL=$MODEL  LORA=$LORA  RETRIEVER=$RETRIEVER  OUTDIR=$OUTDIR"
echo "[p0.20B] COMEM_SELECTOR=$COMEM_SELECTOR (iter_bm25=cross-selector flagship; dense_bge=A-P1.1 same-selector control)"
[ "$RUN" != "1" ] && echo "[p0.20B] *** DRY-RUN (RUN!=1): printing commands only, no forward ***"
if [ "$CALIB_OFFSET" -lt "$LIMIT" ]; then
    echo "[p0.20B][FATAL] CALIB_OFFSET($CALIB_OFFSET) < LIMIT($LIMIT) => calib/quality split overlap."; exit 2
fi
echo "============================================================"

# LOGDIR follows OUTDIR (overridable) so per-OUTDIR runs get their OWN done/ markers.
# Default OUTDIR=bench_results/p0_20_phaseB_dense => logs/p0_20_phaseB_dense (original
# run already complete; its stale done/ is never re-read so the differing basename is harmless).
LOGDIR="${LOGDIR:-logs/$(basename "$OUTDIR")}"
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
    export PYTHONHASHSEED=0
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/scripts:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$PROJECT_ROOT/.hf_cache"
    export http_proxy="" https_proxy="" all_proxy=""
    mkdir -p "$LOGDIR" "$DONEDIR" "$OUTDIR"
fi

# ---- STEP 0: manifest (strict-fix + BGE sha gate). Aborts (exit 3) on mismatch. ----
echo "[p0.20B] STEP 0: manifest / strict-fix + BGE sha gate on GPU 0"
_do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON >\"$LOGDIR/manifest.out\" 2>&1"
if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
    echo "[p0.20B] MANIFEST ABORTED — see $LOGDIR/manifest.out"; tail -20 "$LOGDIR/manifest.out"; exit 3
fi

# ---- STEP 1: sanity gate ----
if [ "$DO_SANITY" = "1" ]; then
    echo "[p0.20B] STEP 1: sanity (LoRA toggle + dense determinism + P1.9 repro + pairing + split) on GPU 0"
    # use a babilong cell for sanity so the P1.9 reproduction cross-check can fire.
    _do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode sanity $COMMON \
--benchmark babilong --task qa1 --length 16k --k 12 >\"$LOGDIR/sanity.out\" 2>&1"
    if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
        echo "[p0.20B] SANITY FAILED — aborting."; tail -25 "$LOGDIR/sanity.out"; exit 4
    fi
fi

# ---- STEP 2: calibration latency sweep (freeze basis; k_dense* chosen HERE) ----
echo "[p0.20B] STEP 2: calibration latency sweep (KS x $N_PROCS procs) on GPU 0 (idle-GPU timing)"
read -r -a KS <<< "$KS_STR"
for k in "${KS[@]}"; do
    for pr in $(seq 0 $((N_PROCS - 1))); do
        _do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode calib_latency $COMMON \
--k $k --proc_id $pr --warmup $WARMUP --n_repeat $N_REPEAT \
--calib_length $CALIB_LENGTH >\"$LOGDIR/calib_k${k}_p${pr}.out\" 2>&1"
    done
done

# ---- STEP 3: build the quality job queue (cells x QKS x shards) ----
echo "[p0.20B] STEP 3: quality job grid (QCELLS x QKS x $NUM_SHARDS shards)"
read -r -a QKS <<< "$QKS_STR"
NJOBS=0
if [ "$RUN" = "1" ]; then : > "$QUEUE"; fi
for cell in $QCELLS; do
    IFS='|' read -r bench task len <<< "$cell"
    for k in "${QKS[@]}"; do
        for sh in $(seq 0 $((NUM_SHARDS - 1))); do
            job="quality|$bench|$task|$len|$k|$sh|$NUM_SHARDS"
            echo "    job: $job"
            [ "$RUN" = "1" ] && echo "$job" >> "$QUEUE"
            NJOBS=$((NJOBS + 1))
        done
    done
done
echo "[p0.20B] queued $NJOBS quality jobs."
FIRST=$(echo $QCELLS | awk '{print $1}')
IFS='|' read -r FB FT FL <<< "$FIRST"
echo "[p0.20B] example per-shard quality command:"
echo "    CUDA_VISIBLE_DEVICES=<gpu> $PYBIN $DRIVER --mode quality $COMMON \\"
echo "        --benchmark $FB --task $FT --length $FL --k 12 --num_shards $NUM_SHARDS --shard_index 0"

# ---- STEP 3b: 8-GPU flock task pool (only when RUN=1) ----
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
            IFS='|' read -r kind bench task len k a b <<< "$job"
            local tag="${kind}_${bench}_${task}_${len}_k${k}_${a}of${b}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            echo "[gpu$gpu] START $job"
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode quality $COMMON \
                --benchmark "$bench" --task "$task" --length "$len" --k "$k" \
                --num_shards "$b" --shard_index "$a" \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[p0.20B] STEP 3b: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[p0.20B] all workers done."

    # ---- STEP 4: aggregate ----
    # --comem_selector MUST be passed here too: the driver derives summary.json's
    # `comem_selector` / `same_selector_control` from this arg, so omitting it
    # falls back to the iter_bm25 default and mislabels a BGE/BGE same-selector
    # arm as the cross-selector flagship (per-cell data stays correct).
    echo "[p0.20B] STEP 4: aggregate"
    $PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" --tol "$TOL" --n_boot "$N_BOOT" \
        --comem_selector "$COMEM_SELECTOR" \
        >"$LOGDIR/aggregate.out" 2>&1
    cat "$LOGDIR/aggregate.out"
    echo "[p0.20B] COMPLETE."
else
    echo "[p0.20B] STEP 3b/4 (workers + aggregate) — DRY: not launched."
    echo "[p0.20B] aggregate (CPU) command MAIN runs after all shards finish:"
    echo "    $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR --tol $TOL --n_boot $N_BOOT --comem_selector $COMEM_SELECTOR"
    echo "[p0.20B] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
