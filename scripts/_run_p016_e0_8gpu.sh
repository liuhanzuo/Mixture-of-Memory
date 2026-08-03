#!/usr/bin/env bash
# ============================================================================
# P0.16 — E0 document-contextual Write control. 4-ARM paired bench (DRY-DEFAULT).
#
# Arm A  : resume_j=0  full 36-layer continuous replay + flagship LoRA (== P0.13 A).
# Arm B  : resume_j=12 upper-24 replay from CHUNK-LOCAL cached h12 + LoRA (== P0.13 B,
#          the DEPLOYABLE endpoint; included by default so E0-vs-deployable is paired
#          in the SAME run; disable with NO_ARMB=1).
# Arm C  : resume_j=12 upper-24 replay from CONTINUOUS pack-level h12 (== P1.7 C,
#          ORACLE, NOT deployable — per-query lower-12 recompute over the pack).
# Arm E0 : resume_j=12 DOCUMENT-CONTEXTUAL Write (new; query-INDEPENDENT, O(L)):
#          layers[0:12] run ONCE continuously/full-causal over the WHOLE document
#          (RoPE 0:N), per-token h12 SLICED at the BM25-selected chunk boundaries
#          into the SAME pack layout as Arm B, read with fresh contiguous pack
#          positions (== Arm B's repositioning). Cross-query-reusable CONTROL, NOT a
#          strict upper bound; its Write is O(L) (write latency + peak reported).
#
# For each example the pack is built ONCE (forward-free iter_bm25 selection,
# resume_j-independent) and ALL FOUR arms read that identical pack => strict 1:1
# pairing. E0's document-contextual layer-12 state == the same-document stock
# lower-12 forward, verified by --mode e0_h12_sanity / --verify (numeric gate).
#
# Attribution (per paperA/TODOList.md §P0.16):
#   E0 ≈ A/C (and E0 >> B) => A-B gap is chunk-independent Write lacking document
#     context; the deployable Read interface / repositioning is largely lossless
#     => prioritise P0.17 (overlap Write to inject context).
#   E0 ≈ B (and E0 << A/C) => even a perfect document-contextual h12 cannot survive
#     repositioning into the selected-pack coords => prioritise P0.18 (2-factor).
#   E0 in between => both factors present => run P0.17 + P0.18.
#
# FLAGSHIP RULER QCMem config (mandated, identical to P0.13 / P1.7 / the headline):
#   selector=iter_bm25  topk=12  iter_hop_topk=4  sink=bos  chunk_size=512
#   chat_template=False  enable_thinking=False  add_bos=0  bf16  sdpa
#   LoRA=outputs/qcmem_distill_qwen_j12_r32_4k/final (sha dd09cd17…, layers 12..35)
#
# Protocol (mandated): niah_multikey_1 x {8k,16k}, n=100/cell, REUSING P1.7 (#121)'s
#   200 paired examples (SAME seed/task/length/chunk_size/selector => same samples;
#   cross-checked via --p013_manifest_dir pack-sha equality). NO new samples.
#
# ★ DRY-BY-DEFAULT: prints the manifest / e0_h12_sanity / quality / aggregate
#   commands for the requested cells but does NOT run any GPU/CPU forward. MAIN
#   launches the real eval by exporting RUN=1 on a FREE diskB H20 node (this harness
#   intentionally does NOT self-run so it never contends with a live training job).
#
# Cohort:
#   COHORT=min (default) = niah_multikey_1 x {16k,8k}   = 2 cells (the mandated set)
#   (override entirely with TASKS="…" LENGTHS="…")
#
# Usage (MAIN, on a FREE diskB H20 node after a GPU frees up):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   COHORT=min RUN=1 setsid nohup bash scripts/_run_p016_e0_8gpu.sh \
#     >logs/p0_16_e0.out 2>&1 &
#   # dry preview (default): omit RUN=1 (or RUN=0) — prints commands only.
# ============================================================================
set -uo pipefail

# ---- overridable env (defaults tuned for diskB; NO auto-run) ----------------
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
# PYTHON_BIN: L20A(wzc1) -> .venv/bin/python ; diskB H20 -> torch-base or .venv.
PYBIN="${PYTHON_BIN:-.venv/bin/python}"
RUN="${RUN:-0}"                      # 0 = DRY (print only) ; 1 = execute
COHORT="${COHORT:-min}"              # min  (or set TASKS / LENGTHS directly)
NUM_SHARDS="${NUM_SHARDS:-4}"        # per-cell shards (each shard n=limit/shards)
LIMIT="${LIMIT:-100}"                # samples per cell
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"  # GPU pool for the flock scheduler
DO_H12_SANITY="${DO_H12_SANITY:-1}"  # gate: verify E0 document-contextual invariant
H12_TOL="${H12_TOL:-5e-2}"           # bf16 max-abs tolerance for the h12 assert
H12_CHECK_PREFIX="${H12_CHECK_PREFIX:-1024}"  # doc-prefix len for the E0 numeric gate
NO_ARMB="${NO_ARMB:-0}"              # 1 => drop the deployable Arm B from the run

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUTDIR="${OUTDIR:-bench_results/p0_16_e0_write_control}"
# cross-check every pack sha against the P1.7 / P0.13 run's per-example JSONL
# (SAME 200 paired examples => identical pack shas; fail-closed if they differ).
P013_MANIFEST_DIR="${P013_MANIFEST_DIR:-bench_results/p1_7_h12_oracle}"
DRIVER="scripts/eval_p016_e0_write_control.py"

ARMB_FLAG=""; [ "$NO_ARMB" = "1" ] && ARMB_FLAG="--no_armB"
COMMON="--model_path $MODEL --lora_adapter $LORA \
--resume_j_a 0 --resume_j_b 12 --resume_j_c 12 --resume_j_e0 12 $ARMB_FLAG \
--topk 12 --iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
--seed 42 --output_dir $OUTDIR"

# ---- cohort -> task/length grid (LPT: longest first) ------------------------
TASKS_DEFAULT="niah_multikey_1"
LENGTHS_DEFAULT="16k 8k"
TASKS="${TASKS:-$TASKS_DEFAULT}"
LENGTHS="${LENGTHS:-$LENGTHS_DEFAULT}"

echo "============================================================"
echo "[p0.16] PROJECT_ROOT=$PROJECT_ROOT"
echo "[p0.16] PYBIN=$PYBIN  RUN=$RUN  COHORT=$COHORT  armB=$([ "$NO_ARMB" = 1 ] && echo off || echo on)"
echo "[p0.16] TASKS=[$TASKS]  LENGTHS=[$LENGTHS]  limit=$LIMIT shards=$NUM_SHARDS"
echo "[p0.16] MODEL=$MODEL  LORA=$LORA  OUTDIR=$OUTDIR"
echo "[p0.16] P013_MANIFEST_DIR=$P013_MANIFEST_DIR  h12_tol=$H12_TOL prefix=$H12_CHECK_PREFIX"
[ "$RUN" != "1" ] && echo "[p0.16] *** DRY-RUN (RUN!=1): printing commands only, no forward ***"
echo "============================================================"

LOGDIR="logs/p0_16_e0"
DONEDIR="$LOGDIR/done"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

# _do CMD: run it (RUN=1) or just print it (dry). All model forwards go through here.
_do() {
    echo "  \$ $*"
    if [ "$RUN" = "1" ]; then
        eval "$@"
        return $?
    fi
    return 0
}

if [ "$RUN" = "1" ]; then
    cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }
    export PYTHONUNBUFFERED=1
    export PYTHONHASHSEED=0           # deterministic hash((task,length)) sample seed
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$PROJECT_ROOT/.hf_cache"
    export http_proxy="" https_proxy="" all_proxy=""
    mkdir -p "$LOGDIR" "$DONEDIR" "$OUTDIR"
fi

# ---- STEP 0: manifest (strict-fix gate). Aborts (exit 3) on hash mismatch. ----
echo "[p0.16] STEP 0: manifest / strict-fix gate on GPU 0"
_do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON >\"$LOGDIR/manifest.out\" 2>&1"
if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
    echo "[p0.16] MANIFEST ABORTED — see $LOGDIR/manifest.out"; tail -20 "$LOGDIR/manifest.out"; exit 3
fi

# ---- STEP 1: E0 document-contextual invariant sanity (E0 validity gate) ----
#   Asserts E0's OWN document-contextual lower-12 forward == stock hidden_states[12]
#   on a document prefix (bf16 max-abs < H12_TOL). If this fails, E0 is invalid.
if [ "$DO_H12_SANITY" = "1" ]; then
    echo "[p0.16] STEP 1: e0_h12_sanity (E0 doc-ctx lower-12 == stock lower-12) on GPU 0"
    _do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode e0_h12_sanity $COMMON \
--task niah_multikey_1 --length 8k --example_index 0 \
--h12_tol $H12_TOL --h12_check_prefix $H12_CHECK_PREFIX \
>\"$LOGDIR/e0_h12_sanity.out\" 2>&1"
    if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
        echo "[p0.16] E0 H12 SANITY FAILED — E0 invalid; aborting."; tail -20 "$LOGDIR/e0_h12_sanity.out"; exit 4
    fi
fi

# ---- STEP 2: build the quality job queue (cells x shards) ----
echo "[p0.16] STEP 2: quality job grid (cells x $NUM_SHARDS shards)"
NJOBS=0
if [ "$RUN" = "1" ]; then : > "$QUEUE"; fi
for len in $LENGTHS; do
    for task in $TASKS; do
        for sh in $(seq 0 $((NUM_SHARDS - 1))); do
            echo "    job: quality|$task|$len|$sh|$NUM_SHARDS"
            [ "$RUN" = "1" ] && echo "quality|$task|$len|$sh|$NUM_SHARDS" >> "$QUEUE"
            NJOBS=$((NJOBS + 1))
        done
    done
done
echo "[p0.16] queued $NJOBS quality jobs."

# per-example command template (dry preview of what each worker runs):
FIRST_TASK=$(echo $TASKS | awk '{print $1}'); FIRST_LEN=$(echo $LENGTHS | awk '{print $1}')
echo "[p0.16] example per-shard command (--verify runs the E0 h12 assert on shard 0's 1st example):"
echo "    CUDA_VISIBLE_DEVICES=<gpu> $PYBIN $DRIVER --mode quality $COMMON \\"
echo "        --task $FIRST_TASK --length $FIRST_LEN --limit $LIMIT \\"
echo "        --num_shards $NUM_SHARDS --shard_index 0 --verify \\"
echo "        --h12_tol $H12_TOL --h12_check_prefix $H12_CHECK_PREFIX \\"
echo "        --p013_manifest_dir $P013_MANIFEST_DIR"

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
            IFS='|' read -r kind task len a b <<< "$job"
            local tag="${kind}_${task}_${len}_${a}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            local verify=""; [ "$a" = "0" ] && verify="--verify"
            echo "[gpu$gpu] START $job"
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode quality $COMMON \
                --task "$task" --length "$len" --limit "$LIMIT" \
                --num_shards "$b" --shard_index "$a" $verify \
                --h12_tol "$H12_TOL" --h12_check_prefix "$H12_CHECK_PREFIX" \
                --p013_manifest_dir "$P013_MANIFEST_DIR" \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[p0.16] STEP 3: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[p0.16] all workers done."

    # ---- STEP 4: aggregate ----
    echo "[p0.16] STEP 4: aggregate"
    $PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" >"$LOGDIR/aggregate.out" 2>&1
    cat "$LOGDIR/aggregate.out"
    echo "[p0.16] COMPLETE."
else
    echo "[p0.16] STEP 3/4 (workers + aggregate) — DRY: not launched."
    echo "[p0.16] aggregate (CPU) command MAIN runs after all shards finish:"
    echo "    $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR"
    echo "[p0.16] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
