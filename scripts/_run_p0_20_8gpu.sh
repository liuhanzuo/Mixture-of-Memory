#!/usr/bin/env bash
# ============================================================================
# P0.20 Stage A — BM25 equal-latency frontier: raw-text RAG vs CoMem.
# 8-GPU flock task-pool scheduler (DRY-BY-DEFAULT). Parallels _run_p017_e2_8gpu.sh.
#
# Research question (paperA §P0.20): at a FIXED online latency budget, can CoMem
# spend the compute it SAVES (skipping the lower j=12 layers, resuming from a cached
# depth-12 residual) to READ MORE evidence chunks and match/beat raw-text RAG?
#
# Two paths, sharing the EXACT SAME retrieved pack per example (forward-free
# iter_bm25 selection is resume_j-independent => bit-identical packs):
#   TEXT-RAG (Config #2): resume_j=0, LoRA DISABLED == vanilla Qwen3-8B full recompute.
#   CoMem    (flagship) : resume_j=12, LoRA ENABLED; online context = pre-stored h12
#                         fetch + H2D (GPU-resident AND CPU-pinned reported separately).
#
# FLAGSHIP config (mandated, identical to P0.13/P1.7/P0.16/Config#2):
#   selector=iter_bm25  iter_hop_topk=4  sink=bos  chunk_size=512
#   chat_template=False  enable_thinking=False  add_bos=0  bf16  sdpa
#   LoRA=outputs/qcmem_distill_qwen_j12_r32_4k/final (sha dd09cd17…, layers 12..35)
#
# Pipeline (all forwards through _do => RUN=1 executes, RUN=0/unset only prints):
#   STEP 0  manifest  — strict-fix gate (backbone+LoRA sha, layers [12..35], 168 mods).
#                       Aborts (exit 3) on mismatch.
#   STEP 1  sanity    — fail-closed gates: LoRA sha; disable_adapter() toggles LoRA;
#                       both arms consume a bit-identical pack (read_len equality);
#                       calibration/quality split disjoint. Aborts (exit 4) on fail.
#   STEP 2  CALIB LATENCY sweep — for each k in KS, on RESERVED calib docs (indices
#                       >= CALIB_OFFSET >= LIMIT), N_REPEAT timed reps x N_PROCS
#                       independent procs: TTFT component breakdown for BOTH arms
#                       (selection + write + read + store-fetch + H2D). This is what
#                       k_RAG*/k_CoMem* are FROZEN on — latency ONLY, never quality.
#   STEP 3  QUALITY pool — 8-GPU flock task-pool over (benchmark,task,length,k) x shards.
#                       Both arms per example on the identical pack. Primary tasks:
#                       BABILong qa1/qa2, LongEval, LoCoMo; secondary: RULER multikey.
#   STEP 4  aggregate — freeze k* (both anchors), build the k-frontier, paired
#                       bootstrap 95% CI + exact McNemar at anchors, success-criterion
#                       verdict. Negative/marginal results reported in full.
#
# ★ DRY-BY-DEFAULT: prints every command but runs NO forward unless RUN=1. MAIN
#   launches the real eval by exporting RUN=1 on a FREE diskB H20 node (.104) so this
#   never contends with a live training/eval job.
#
# Usage (MAIN, on FREE diskB .104 after its GPUs free):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_p0_20_8gpu.sh >logs/p0_20_eqlat.out 2>&1 &
#   # dry preview (default): omit RUN=1 — prints commands only.
# ============================================================================
set -uo pipefail

# ---- overridable env (defaults tuned for diskB; NO auto-run) ----------------
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYBIN="${PYTHON_BIN:-.venv/bin/python}"      # L20A(wzc1)->.venv ; diskB H20->torch-base or .venv
RUN="${RUN:-0}"                              # 0 = DRY (print only) ; 1 = execute
NUM_SHARDS="${NUM_SHARDS:-4}"                # per-cell shards (each shard n=limit/shards)
LIMIT="${LIMIT:-100}"                        # quality samples per cell
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"
DO_SANITY="${DO_SANITY:-1}"

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUTDIR="${OUTDIR:-bench_results/p0_20_eqlat}"
DRIVER="scripts/eval_p0_20_equal_latency.py"

# ---- k sweep (BOTH arms) + calibration latency reps -------------------------
KS_STR="${KS:-2 4 6 8 10 12 14 16 20 24}"    # retrieved-chunk grid
N_PROCS="${N_PROCS:-3}"                       # >=3 independent latency procs (mandated)
WARMUP="${WARMUP:-5}"
N_REPEAT="${N_REPEAT:-20}"                     # >=20 timed reps/point (mandated)
CALIB_OFFSET="${CALIB_OFFSET:-900}"           # reserved calib base idx (MUST be >= LIMIT)
CALIB_LENGTH="${CALIB_LENGTH:-32k}"           # >= 24 ctx chunks so k<=24 fits
TOL="${TOL:-0.05}"                            # +/-5% pre-registered latency band
N_BOOT="${N_BOOT:-10000}"

# ---- quality grid: primary=BABILong qa1/qa2 + LongEval + LoCoMo; secondary=RULER
# a "cell" = benchmark|task|length. Defaults are the mandated primary+secondary set.
# Override with QCELLS="bench|task|length …". k sweep is applied to every cell.
QCELLS_DEFAULT="babilong|qa1|4k babilong|qa1|16k babilong|qa2|4k babilong|qa2|16k longeval|longeval|8k longeval|longeval|16k locomo|locomo|na ruler|niah_multikey_1|8k ruler|niah_multikey_1|16k"
QCELLS="${QCELLS:-$QCELLS_DEFAULT}"
# quality-side k grid. DEFAULT = full sweep KS so whatever k_RAG*/k_CoMem* the
# calibration freezes (any integer in KS) HAS a matching quality cell for the
# anchor comparison. Subset via QKS="…" for a faster first pass (frontier only).
QKS_STR="${QKS:-$KS_STR}"

COMMON="--model_path $MODEL --lora_adapter $LORA \
--iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
--seed 42 --output_dir $OUTDIR --limit $LIMIT --calib_offset $CALIB_OFFSET"

echo "============================================================"
echo "[p0.20] PROJECT_ROOT=$PROJECT_ROOT"
echo "[p0.20] PYBIN=$PYBIN  RUN=$RUN"
echo "[p0.20] KS=[$KS_STR]  QKS=[$QKS_STR]  n_procs=$N_PROCS warmup=$WARMUP n_repeat=$N_REPEAT tol=$TOL"
echo "[p0.20] QCELLS=[$QCELLS]"
echo "[p0.20] LIMIT=$LIMIT shards=$NUM_SHARDS calib_offset=$CALIB_OFFSET calib_length=$CALIB_LENGTH"
echo "[p0.20] MODEL=$MODEL  LORA=$LORA  OUTDIR=$OUTDIR"
[ "$RUN" != "1" ] && echo "[p0.20] *** DRY-RUN (RUN!=1): printing commands only, no forward ***"
if [ "$CALIB_OFFSET" -lt "$LIMIT" ]; then
    echo "[p0.20][FATAL] CALIB_OFFSET($CALIB_OFFSET) < LIMIT($LIMIT) => calib/quality split overlap."; exit 2
fi
echo "============================================================"

# LOGDIR follows OUTDIR (overridable) so per-OUTDIR runs get their OWN done/ markers.
# Backward-compat: default OUTDIR=bench_results/p0_20_eqlat => logs/p0_20_eqlat (== old value).
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

# ---- STEP 0: manifest (strict-fix gate). Aborts (exit 3) on hash mismatch. ----
echo "[p0.20] STEP 0: manifest / strict-fix gate on GPU 0"
_do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON >\"$LOGDIR/manifest.out\" 2>&1"
if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
    echo "[p0.20] MANIFEST ABORTED — see $LOGDIR/manifest.out"; tail -20 "$LOGDIR/manifest.out"; exit 3
fi

# ---- STEP 1: sanity gate (LoRA toggle + pack pairing + split isolation) ----
if [ "$DO_SANITY" = "1" ]; then
    echo "[p0.20] STEP 1: sanity (LoRA sha + disable_adapter toggle + pack pairing + split) on GPU 0"
    _do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode sanity $COMMON \
--benchmark ruler --task niah_multikey_1 --length 8k --k 12 >\"$LOGDIR/sanity.out\" 2>&1"
    if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
        echo "[p0.20] SANITY FAILED — aborting."; tail -25 "$LOGDIR/sanity.out"; exit 4
    fi
fi

# ---- STEP 2: calibration latency sweep (freeze basis; k* chosen HERE, latency-only) ----
# One process per (k, proc_id). N_PROCS independent procs per k -> median across procs.
# Run these sequentially on GPU 0 so timings are not perturbed by co-resident jobs
# (latency measurement must be on an otherwise-idle GPU).
echo "[p0.20] STEP 2: calibration latency sweep (KS x $N_PROCS procs) on GPU 0 (idle-GPU timing)"
read -r -a KS <<< "$KS_STR"
for k in "${KS[@]}"; do
    for pr in $(seq 0 $((N_PROCS - 1))); do
        _do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode calib_latency $COMMON \
--k $k --proc_id $pr --warmup $WARMUP --n_repeat $N_REPEAT \
--calib_length $CALIB_LENGTH >\"$LOGDIR/calib_k${k}_p${pr}.out\" 2>&1"
    done
done

# ---- STEP 3: build the quality job queue (cells x QKS x shards) ----
echo "[p0.20] STEP 3: quality job grid (QCELLS x QKS x $NUM_SHARDS shards)"
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
echo "[p0.20] queued $NJOBS quality jobs."
FIRST=$(echo $QCELLS | awk '{print $1}')
IFS='|' read -r FB FT FL <<< "$FIRST"
echo "[p0.20] example per-shard quality command:"
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
    echo "[p0.20] STEP 3b: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[p0.20] all workers done."

    # ---- STEP 4: aggregate ----
    echo "[p0.20] STEP 4: aggregate"
    $PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" --tol "$TOL" --n_boot "$N_BOOT" \
        >"$LOGDIR/aggregate.out" 2>&1
    cat "$LOGDIR/aggregate.out"
    echo "[p0.20] COMPLETE."
else
    echo "[p0.20] STEP 3b/4 (workers + aggregate) — DRY: not launched."
    echo "[p0.20] aggregate (CPU) command MAIN runs after all shards finish:"
    echo "    $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR --tol $TOL --n_boot $N_BOOT"
    echo "[p0.20] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
