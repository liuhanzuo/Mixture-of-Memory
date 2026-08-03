#!/usr/bin/env bash
# ============================================================================
# P0.18 — E4 two-factor (2x2) decomposition of the deployable Write gap.
# 5-ARM paired bench (DRY-DEFAULT). Splits the P0.16 "deployable gap" (A-BB) into
# factor1 (lower-layer attention scope) x factor2 (read position IDs).
#
#   A  : resume_j=0  full 36-layer continuous replay + flagship LoRA (== P0.13/P1.7/
#        P0.16 Arm A). The paired logit-KL / top1 anchor.
#   BB : resume_j=12 chunk-local h12 read at LOCAL/CONTIGUOUS pack positions
#        (== P0.16/P0.13 Arm B; the DEPLOYABLE endpoint). Run VERBATIM through
#        p017._run_arm => bit-identical to the headline row.
#   E0 : resume_j=12 DOCUMENT-CONTEXTUAL h12 (layers[0:12] run once, full-causal over
#        the whole doc) read at LOCAL/CONTIGUOUS positions (== P0.16 E0). Run VERBATIM
#        through p016._run_e0. Disable with NO_E0=1 (then E0-based controls unavailable).
#   X  : resume_j=12 CHUNK-LOCAL h12 read at DOCUMENT-ORIGIN RoPE positions (NEW).
#        Isolates factor2 (read repositioning cost) at fixed factor1.
#   Y  : resume_j=12 DOCUMENT-CONTEXTUAL h12 read at DOCUMENT-ORIGIN RoPE positions
#        (NEW; "no repositioning at all" corner).
#
# 2x2 factors:
#   factor1 (lower-layer scope):  chunk-local  vs  document-contextual
#   factor2 (read position IDs):  local/reset  vs  document-origin
#   arms:   BB=(cl,local)  E0=(dc,local)  X=(cl,docpos)  Y=(dc,docpos)  [+ A anchor]
#
# Single-factor controls (each flips EXACTLY one factor):
#   BB->E0 : factor1 (lower-layer document context value)      [needs E0 on]
#   BB->X  : factor2 (store->read RoPE repositioning cost)
#   E0->Y  : factor2 at doc-ctx h12                            [needs E0 on]
#   X ->Y  : factor1 at doc-origin coords
#   interaction: (BB->E0)+(BB->X) vs (BB->Y). Non-additive => factors interact
#                (must be fixed jointly). Any sign admissible; paired CI + McNemar
#                is the deliverable (=> P0.17 overlap-Write if factor1 dominates;
#                => P1.10 learned position interface if factor2 dominates).
#
# ★ SAFETY OF DOCUMENT-ORIGIN READ (why none of the 4 arms is dropped): document-origin
#   RoPE is applied via rotary_emb(position_ids=doc_origin) while the ATTENTION MASK is
#   built from CONTIGUOUS positions, so find_packed_sequence_indices does NOT split the
#   pack into a block-diagonal mask. The read stays full-causal; only the RoPE coord
#   moves — the single declared factor2 change, nothing else. Validated by
#   --mode pos_sanity (custom read w/ contiguous positions == qc.read_prefill to fp tol)
#   AND per-run by --verify (STEP 1b runs the same plumbing gate on shard0's 1st ex).
#
# For each example the pack is built ONCE (forward-free iter_bm25, resume_j-independent)
# and ALL arms read that identical pack => strict 1:1 pairing (cross-checked vs the
# P1.7 pack shas via --p013_manifest_dir).
#
# FLAGSHIP RULER QCMem config (mandated, identical to P0.13/P1.7/P0.16/the headline):
#   selector=iter_bm25 topk=12 iter_hop_topk=4 sink=bos chunk_size=512
#   chat_template=False enable_thinking=False add_bos=0 bf16 sdpa
#   LoRA=outputs/qcmem_distill_qwen_j12_r32_4k/final (sha dd09cd17…, layers 12..35)
#
# Protocol (mandated): niah_multikey_1 x {8k,16k}, n=100/cell, REUSING P1.7 (#121)'s
#   200 paired examples (SAME seed/task/length/chunk_size/selector => same samples).
#
# ★ DRY-BY-DEFAULT: prints the manifest / pos_sanity / quality / aggregate commands but
#   runs NO GPU/CPU forward. MAIN launches the real eval with RUN=1 on a FREE diskB H20
#   node (this harness never self-runs, so it can't contend with a live training job).
#
# Cohort:
#   COHORT=min (default) = niah_multikey_1 x {16k,8k} = 2 cells (the mandated set)
#   (override entirely with TASKS="…" LENGTHS="…")
#
# Usage (MAIN, on a FREE diskB H20 node after a GPU frees up):
#   PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=.venv/bin/python \
#   COHORT=min RUN=1 setsid nohup bash scripts/_run_p018_e4_8gpu.sh \
#     >logs/p0_18_e4.out 2>&1 &
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
DO_POS_SANITY="${DO_POS_SANITY:-1}"  # gate: doc-origin read plumbing == read_prefill
DO_H12_SANITY="${DO_H12_SANITY:-1}"  # gate (via --verify): doc-ctx h12 == stock l12
H12_TOL="${H12_TOL:-5e-2}"           # bf16 max-abs tolerance for the doc-ctx h12 assert
H12_CHECK_PREFIX="${H12_CHECK_PREFIX:-1024}"  # doc-prefix len for the doc-ctx gate
POS_TOL="${POS_TOL:-5e-2}"           # bf16 max-abs tolerance for the pos-plumbing assert
NO_E0="${NO_E0:-0}"                  # 1 => drop the E0 (dc,local) arm; keep A/BB/X/Y

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUTDIR="${OUTDIR:-bench_results/p0_18_e4_2x2}"
# cross-check every pack sha against the P1.7 / P0.13 / P0.16 run's per-example JSONL.
P013_MANIFEST_DIR="${P013_MANIFEST_DIR:-bench_results/p1_7_h12_oracle}"
DRIVER="scripts/eval_p018_e4_2x2_writecontrol.py"

E0_FLAG=""; [ "$NO_E0" = "1" ] && E0_FLAG="--no_e0"
COMMON="--model_path $MODEL --lora_adapter $LORA \
--resume_j_a 0 --resume_j 12 $E0_FLAG \
--topk 12 --iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
--seed 42 --output_dir $OUTDIR"

# ---- cohort -> task/length grid (LPT: longest first) ------------------------
TASKS_DEFAULT="niah_multikey_1"
LENGTHS_DEFAULT="16k 8k"
TASKS="${TASKS:-$TASKS_DEFAULT}"
LENGTHS="${LENGTHS:-$LENGTHS_DEFAULT}"

echo "============================================================"
echo "[p0.18] PROJECT_ROOT=$PROJECT_ROOT"
echo "[p0.18] PYBIN=$PYBIN  RUN=$RUN  COHORT=$COHORT  e0=$([ "$NO_E0" = 1 ] && echo off || echo on)"
echo "[p0.18] TASKS=[$TASKS]  LENGTHS=[$LENGTHS]  limit=$LIMIT shards=$NUM_SHARDS"
echo "[p0.18] MODEL=$MODEL  LORA=$LORA  OUTDIR=$OUTDIR"
echo "[p0.18] P013_MANIFEST_DIR=$P013_MANIFEST_DIR  h12_tol=$H12_TOL pos_tol=$POS_TOL prefix=$H12_CHECK_PREFIX"
[ "$RUN" != "1" ] && echo "[p0.18] *** DRY-RUN (RUN!=1): printing commands only, no forward ***"
echo "============================================================"

LOGDIR="logs/p0_18_e4"
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
echo "[p0.18] STEP 0: manifest / strict-fix gate on GPU 0"
_do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON >\"$LOGDIR/manifest.out\" 2>&1"
if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
    echo "[p0.18] MANIFEST ABORTED — see $LOGDIR/manifest.out"; tail -20 "$LOGDIR/manifest.out"; exit 3
fi

# ---- STEP 1a: pos_sanity — doc-origin read plumbing == read_prefill (X/Y validity) ----
#   Feeds CONTIGUOUS positions through the CUSTOM doc-origin read path; the first-step
#   logits must match qc.read_prefill (Arm B path) to POS_TOL. This proves the custom
#   path changes ONLY the RoPE coordinate (no block-diagonal artifact), so X/Y isolate
#   factor2 cleanly. If it fails, the doc-origin arms are invalid.
if [ "$DO_POS_SANITY" = "1" ]; then
    echo "[p0.18] STEP 1a: pos_sanity (custom docpos read w/ contiguous == read_prefill) on GPU 0"
    _do "CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode pos_sanity $COMMON \
--task niah_multikey_1 --length 8k --example_index 0 --pos_tol $POS_TOL \
>\"$LOGDIR/pos_sanity.out\" 2>&1"
    if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
        echo "[p0.18] POS_SANITY FAILED — doc-origin read invalid; aborting."; tail -20 "$LOGDIR/pos_sanity.out"; exit 4
    fi
fi

# NOTE: the E0/Y document-contextual h12 invariant (== stock lower-12) is checked
# INLINE on shard0's first example via --verify (STEP 3), reusing P0.16's numeric gate.

# ---- STEP 2: build the quality job queue (cells x shards) ----
echo "[p0.18] STEP 2: quality job grid (cells x $NUM_SHARDS shards)"
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
echo "[p0.18] queued $NJOBS quality jobs."

# per-example command template (dry preview of what each worker runs):
FIRST_TASK=$(echo $TASKS | awk '{print $1}'); FIRST_LEN=$(echo $LENGTHS | awk '{print $1}')
echo "[p0.18] example per-shard command (--verify runs BOTH the doc-ctx h12 assert AND"
echo "         the doc-pos read-plumbing assert on shard 0's 1st example):"
echo "    CUDA_VISIBLE_DEVICES=<gpu> $PYBIN $DRIVER --mode quality $COMMON \\"
echo "        --task $FIRST_TASK --length $FIRST_LEN --limit $LIMIT \\"
echo "        --num_shards $NUM_SHARDS --shard_index 0 --verify \\"
echo "        --h12_tol $H12_TOL --h12_check_prefix $H12_CHECK_PREFIX --pos_tol $POS_TOL \\"
echo "        --p013_manifest_dir $P013_MANIFEST_DIR"

# ---- STEP 3: 8-GPU flock task pool (only when RUN=1) ----
if [ "$RUN" = "1" ]; then
    read -r -a GPUS <<< "$GPUS_STR"
    VERIFY_ARGS=""; [ "$DO_H12_SANITY" = "1" ] && VERIFY_ARGS="--verify"
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
            local verify=""; [ "$a" = "0" ] && verify="$VERIFY_ARGS"
            echo "[gpu$gpu] START $job"
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode quality $COMMON \
                --task "$task" --length "$len" --limit "$LIMIT" \
                --num_shards "$b" --shard_index "$a" $verify \
                --h12_tol "$H12_TOL" --h12_check_prefix "$H12_CHECK_PREFIX" \
                --pos_tol "$POS_TOL" \
                --p013_manifest_dir "$P013_MANIFEST_DIR" \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[p0.18] STEP 3: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[p0.18] all workers done."

    # ---- STEP 4: aggregate ----
    echo "[p0.18] STEP 4: aggregate"
    $PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" >"$LOGDIR/aggregate.out" 2>&1
    cat "$LOGDIR/aggregate.out"
    echo "[p0.18] COMPLETE."
else
    echo "[p0.18] STEP 3/4 (workers + aggregate) — DRY: not launched."
    echo "[p0.18] aggregate (CPU) command MAIN runs after all shards finish:"
    echo "    $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR"
    echo "[p0.18] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
