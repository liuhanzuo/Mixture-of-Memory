#!/usr/bin/env bash
# ============================================================================
# P1.9 — Dense-retriever + native-prompting standard RAG reference (DRY-DEFAULT).
#
# A ZERO-TRAINING reference: swaps ONLY the selector of the config#2 j0-RAG
# reader — lexical iter_bm25 -> a FROZEN public DENSE retriever (BGE-large-en-v1.5,
# CLS+L2+cosine) — while keeping the reader byte-identical (Qwen3-8B, NO LoRA,
# resume_j=0 full-depth recompute, sink=bos, chunk=512, bf16+sdpa, seed 42) and
# the EXAMPLES identical (each family's own unmodified sample builder + seed/shard
# convention => example i here == example i in BM25 j=0 and CoMem => 1:1 pairing).
#
# Does NOT replace matched-BM25 j=0 (config#2) and is NOT conflated with MemoryLLM.
#
# Report decomposition per cell (--mode aggregate):
#   recall@k (gold support chunk in dense top-k, answer-INDEPENDENT) + Wilson CI,
#   reader acc CONDITIONAL-ON-HIT / -ON-MISS, end-to-end quality, retrieval latency,
#   index size. LoCoMo also emits F1 + judge fields for an offline GPT-4o pass.
#
# ★ MUST report ALL requested tasks (the aggregate has an all-tasks fail-closed
#   guard, exit 5) — forbidden to show only dense-wins tasks.
#
# Two reader-prompt protocols (both zero-shot, greedy):
#   plain  (default) = unified no-chat main protocol (chat_template OFF; the exact
#                      config#2 j0-RAG口径, add_special_tokens=True).
#   native           = reader native-prompt / template-sensitivity variant
#                      (chat_template ON, no-think generation boundary).
#   Set READER_PROMPTS="plain native" to run both.
#
# ★ DRY-BY-DEFAULT: prints the provenance / per-shard / aggregate commands for the
#   requested cells but runs NO GPU forward. MAIN launches the real eval by
#   exporting RUN=1 on a FREE diskB H20 node (this harness never self-contends
#   with a live training job).
#
# Cohort (COHORT=min default = the mandated priority set):
#   babilong qa1/qa2 x {4k,8k,16k}  +  longeval {8k,16k}  +  locomo (single)
#   +  ruler niah_multikey_1 x {8k,16k}   (RULER attached, NOT the sole conclusion)
#   Override entirely with BABI_TASKS / BABI_LENGTHS / LE_LENGTHS / RULER_TASKS /
#   RULER_LENGTHS / DO_LOCOMO=0.
#
# Usage (MAIN, on a FREE diskB H20 node — e.g. .104):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_p1_9_dense_rag_8gpu.sh \
#     >logs/p1_9_dense_rag.out 2>&1 &
#   # dry preview (default): omit RUN=1 — prints commands only.
# ============================================================================
set -uo pipefail

# ---- overridable env (defaults tuned for wzc1; NO auto-run) -----------------
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
# PYTHON_BIN: L20A(wzc1) -> .venv/bin/python ; diskB .104 -> torch-base (.venv broken).
PYBIN="${PYTHON_BIN:-.venv/bin/python}"
RUN="${RUN:-0}"                        # 0 = DRY (print only) ; 1 = execute
COHORT="${COHORT:-min}"
NUM_SHARDS="${NUM_SHARDS:-4}"          # per-cell shards (each shard n=limit/shards)
LIMIT="${LIMIT:-100}"                  # samples per cell
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"    # GPU pool for the flock scheduler
READER_PROMPTS="${READER_PROMPTS:-plain}"  # "plain" | "native" | "plain native"

MODEL="${MODEL:-models/Qwen3-8b-local}"
RETRIEVER="${RETRIEVER:-models/bge-large-en-v1.5}"
TOPK="${TOPK:-12}"
CHUNK="${CHUNK:-512}"
MAXNEW="${MAXNEW:-48}"
OUTDIR="${OUTDIR:-bench_results/p1_9_dense_rag}"
INDEXDIR="${INDEXDIR:-retrieval_results/p1_9_dense}"
LOCOMO_DATA="${LOCOMO_DATA:-locomo/data/locomo10.json}"
DRIVER="scripts/eval_p1_9_dense_rag.py"

# reader is training-free full-depth: NO LoRA, resume_j=0 (hard-checked in driver).
COMMON="--model_path $MODEL --retriever_path $RETRIEVER \
--resume_j 0 --topk $TOPK --chunk_size $CHUNK --sink_tokens bos \
--max_new_tokens $MAXNEW --dtype bfloat16 --attn_impl sdpa --seed 42 \
--limit $LIMIT --output_dir $OUTDIR --index_dir $INDEXDIR \
--locomo_data $LOCOMO_DATA"

# ---- cohort -> per-family task/length grids ---------------------------------
BABI_TASKS="${BABI_TASKS:-qa1 qa2}"
BABI_LENGTHS="${BABI_LENGTHS:-16k 8k 4k}"
LE_LENGTHS="${LE_LENGTHS:-16k 8k}"
RULER_TASKS="${RULER_TASKS:-niah_multikey_1}"
RULER_LENGTHS="${RULER_LENGTHS:-16k 8k}"
DO_LOCOMO="${DO_LOCOMO:-1}"

# require_family spec for the aggregate all-tasks guard (fail-closed):
REQUIRE="babilong:$(echo $BABI_TASKS | tr ' ' ',') longeval: ruler:$(echo $RULER_TASKS | tr ' ' ',')"
[ "$DO_LOCOMO" = "1" ] && REQUIRE="$REQUIRE locomo:"

echo "============================================================"
echo "[p1.9] PROJECT_ROOT=$PROJECT_ROOT"
echo "[p1.9] PYBIN=$PYBIN  RUN=$RUN  COHORT=$COHORT  reader_prompts=[$READER_PROMPTS]"
echo "[p1.9] reader=$MODEL (NO LoRA, resume_j=0)  retriever=$RETRIEVER (BGE CLS+L2+cosine)"
echo "[p1.9] topk=$TOPK chunk=$CHUNK limit=$LIMIT shards=$NUM_SHARDS"
echo "[p1.9] babilong=[$BABI_TASKS] x [$BABI_LENGTHS]  longeval=[$LE_LENGTHS]  ruler=[$RULER_TASKS] x [$RULER_LENGTHS]  locomo=$DO_LOCOMO"
echo "[p1.9] OUTDIR=$OUTDIR  INDEXDIR=$INDEXDIR"
[ "$RUN" != "1" ] && echo "[p1.9] *** DRY-RUN (RUN!=1): printing commands only, no forward ***"
echo "============================================================"

LOGDIR="logs/p1_9_dense_rag"
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
    export PYTHONHASHSEED=0             # deterministic per-(task,length) sample seed
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$PROJECT_ROOT/.hf_cache"
    export http_proxy="" https_proxy="" all_proxy=""
    mkdir -p "$LOGDIR" "$DONEDIR" "$OUTDIR" "$INDEXDIR"
fi

# ---- STEP 0: retriever provenance fail-closed gate (BGE weight sha256) ----
echo "[p1.9] STEP 0: retriever provenance gate (BGE-large-en-v1.5 sha256) on CPU"
_do "$PYBIN $DRIVER --mode provenance --retriever_path $RETRIEVER >\"$LOGDIR/provenance.out\" 2>&1"
if [ "$RUN" = "1" ] && [ $? -ne 0 ]; then
    echo "[p1.9] PROVENANCE GATE FAILED — retriever sha mismatch; aborting."
    tail -20 "$LOGDIR/provenance.out"; exit 6
fi

# ---- STEP 1/2: build the (family,task,length,shard,prompt) job queue ----
echo "[p1.9] STEP 1/2: job grid (cells x $NUM_SHARDS shards x [$READER_PROMPTS])"
NJOBS=0
if [ "$RUN" = "1" ]; then : > "$QUEUE"; fi
_emit() {  # family task length
    local fam="$1" task="$2" len="$3" rp sh
    for rp in $READER_PROMPTS; do
        for sh in $(seq 0 $((NUM_SHARDS - 1))); do
            echo "    job: $fam|$task|$len|$sh|$NUM_SHARDS|$rp"
            [ "$RUN" = "1" ] && echo "$fam|$task|$len|$sh|$NUM_SHARDS|$rp" >> "$QUEUE"
            NJOBS=$((NJOBS + 1))
        done
    done
}
# longest lengths first (LPT-ish: heavy cells drain into the pool early)
for len in $BABI_LENGTHS;  do for t in $BABI_TASKS;  do _emit babilong "$t" "$len"; done; done
for len in $RULER_LENGTHS; do for t in $RULER_TASKS; do _emit ruler    "$t" "$len"; done; done
for len in $LE_LENGTHS;    do _emit longeval NONE "$len"; done
[ "$DO_LOCOMO" = "1" ] && _emit locomo NONE all
echo "[p1.9] queued $NJOBS jobs."

# per-shard command preview (what each worker runs):
echo "[p1.9] example per-shard command:"
echo "    CUDA_VISIBLE_DEVICES=<gpu> $PYBIN $DRIVER --mode run $COMMON \\"
echo "        --family babilong --task qa1 --length 8k --reader_prompt plain \\"
echo "        --num_shards $NUM_SHARDS --shard_index 0 --device cuda:0"

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
            IFS='|' read -r fam task len a b rp <<< "$job"
            local targ=""; [ "$task" != "NONE" ] && targ="--task $task"
            local tag="${fam}_${task}_${len}_${rp}_${a}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            echo "[gpu$gpu] START $job"
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode run $COMMON \
                --family "$fam" $targ --length "$len" --reader_prompt "$rp" \
                --num_shards "$b" --shard_index "$a" --device cuda:0 \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[p1.9] STEP 3: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[p1.9] all workers done."

    # ---- STEP 4: aggregate (all-tasks fail-closed guard, exit 5 if missing) ----
    echo "[p1.9] STEP 4: aggregate + all-tasks guard"
    $PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" \
        --require_family $REQUIRE >"$LOGDIR/aggregate.out" 2>&1
    AGG_RC=$?
    cat "$LOGDIR/aggregate.out"
    [ $AGG_RC -ne 0 ] && { echo "[p1.9] AGGREGATE GUARD FAILED rc=$AGG_RC"; exit $AGG_RC; }
    echo "[p1.9] COMPLETE."
else
    echo "[p1.9] STEP 3/4 (workers + aggregate) — DRY: not launched."
    echo "[p1.9] aggregate (CPU) command MAIN runs after all shards finish:"
    echo "    $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR --require_family $REQUIRE"
    echo "[p1.9] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
