#!/usr/bin/env bash
# ============================================================================
# #144 CoMem dense-selector swap (Paper A) — 8-GPU flock task pool (DRY-DEFAULT).
#
# SINGLE-VARIABLE control vs flagship CoMem: the ONLY changed variable is the
# SELECTOR — flagship lexical iter_bm25 -> a FROZEN public DENSE retriever
# (BGE-large-en-v1.5, CLS+L2+cosine). Everything else is held byte-identical to
# flagship CoMem: reader = models/Qwen3-8b-local + flagship LoRA
# (outputs/qcmem_distill_qwen_j12_r32_4k/final, sha-gated), resume_j=12, adapter
# ENABLED, sink=bos, chunk_size=512, topk=12, bf16 + sdpa, seed 42,
# chat_template=False (BASE LM — no SFT/RL so a chat template is unfair; do NOT
# pass --reader_prompt native), enable_thinking=False, n=100/cell. The dense
# top-12 doc-absolute chunk indices are fed as the oracle needle_chunk_set into
# the UNMODIFIED qcmem_generate -> they become CoMem's h12.
#
# Eval targets (mirror #143 CacheBlend so the two baselines are 1:1 comparable):
#   RULER  : {niah_single_2, niah_multikey_1, variable_tracking} x {4k,8k,16k,32k}
#            (RULER Cohort A)                       -> via eval_p1_9 iter_ruler
#   LoCoMo : full                                   -> via eval_p1_9 iter_locomo
#   BABILong: qa5 x {0k,1k,2k,4k,8k,16k}            -> via eval_p1_9 iter_babilong
#
# 8-GPU flock task pool (same pattern as _run_p1_9_dense_rag_8gpu.sh): one shared
# queue of (family|task|length|shard) jobs; whichever GPU is free pops the next
# job -> dynamic load balancing (long 32k / LoCoMo cells never starve a GPU).
#
# Usage (MAIN, on a FREE diskB H20 node — e.g. .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_comem_dense_selector_8gpu.sh \
#     >logs/dense_selector/sched.out 2>&1 &
#   # dry preview (default): omit RUN=1 — prints commands + job count, no forward.
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
RUN="${RUN:-0}"                           # 0 = DRY (print only) ; 1 = execute

MODEL="${MODEL:-models/Qwen3-8b-local}"
RETRIEVER="${RETRIEVER:-models/bge-large-en-v1.5}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"   # flagship CoMem LoRA (sha-gated)
RESUME_J="${RESUME_J:-12}"
TOPK="${TOPK:-12}"
CHUNK="${CHUNK:-512}"
MAXNEW="${MAXNEW:-48}"
LIMIT="${LIMIT:-100}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SEED="${SEED:-42}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"
READER_PROMPT="${READER_PROMPT:-plain}"   # plain = chat_template OFF (BASE LM)

# per-benchmark task/length grids (override to shrink a preview run).
RULER_TASKS="${RULER_TASKS:-niah_single_2 niah_multikey_1 variable_tracking}"
RULER_LENGTHS="${RULER_LENGTHS:-32k 16k 8k 4k}"   # longest first (LPT)
BABI_TASKS="${BABI_TASKS:-qa5}"
BABI_LENGTHS="${BABI_LENGTHS:-16k 8k 4k 2k 1k 0k}"
DO_LOCOMO="${DO_LOCOMO:-1}"

OUTDIR="${OUTDIR:-bench_results/dense_selector}"
INDEXDIR="${INDEXDIR:-retrieval_results/dense_selector}"
LOGDIR="${LOGDIR:-logs/dense_selector}"
LOCOMO_DATA="${LOCOMO_DATA:-locomo/data/locomo10.json}"
DRIVER="scripts/eval_comem_dense_selector.py"

COMMON="--model_path $MODEL --retriever_path $RETRIEVER --lora_adapter $LORA \
--resume_j $RESUME_J --topk $TOPK --chunk_size $CHUNK --sink_tokens bos \
--max_new_tokens $MAXNEW --dtype bfloat16 --attn_impl sdpa --seed $SEED \
--limit $LIMIT --reader_prompt $READER_PROMPT --output_dir $OUTDIR \
--index_dir $INDEXDIR --locomo_data $LOCOMO_DATA"

# require_family spec for the aggregate all-tasks guard (fail-closed):
REQUIRE="babilong:$(echo $BABI_TASKS | tr ' ' ',') ruler:$(echo $RULER_TASKS | tr ' ' ',')"
[ "$DO_LOCOMO" = "1" ] && REQUIRE="$REQUIRE locomo:"

echo "============================================================"
echo "[#144] PROJECT_ROOT=$PROJECT_ROOT  PYBIN=$PYBIN  RUN=$RUN"
echo "[#144] reader=$MODEL + LoRA=$LORA (resume_j=$RESUME_J, adapter ENABLED)"
echo "[#144] selector=dense_bge ($RETRIEVER, CLS+L2+cosine) topk=$TOPK chunk=$CHUNK"
echo "[#144] reader_prompt=$READER_PROMPT (chat_template OFF)  limit=$LIMIT shards=$NUM_SHARDS seed=$SEED"
echo "[#144] RULER tasks=[$RULER_TASKS] lengths=[$RULER_LENGTHS]"
echo "[#144] BABILong tasks=[$BABI_TASKS] lengths=[$BABI_LENGTHS]  LoCoMo=$DO_LOCOMO"
echo "[#144] OUTDIR=$OUTDIR  INDEXDIR=$INDEXDIR  LOGDIR=$LOGDIR"
[ "$RUN" != "1" ] && echo "[#144] *** DRY-RUN (RUN!=1): commands only, no forward ***"
echo "============================================================"

QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"
DONEDIR="$LOGDIR/done"

if [ "$RUN" = "1" ]; then
    cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }
    export PYTHONUNBUFFERED=1
    export PYTHONHASHSEED=0
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/scripts:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
    export WANDB_MODE=offline
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$PROJECT_ROOT/.hf_cache" HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
    export http_proxy="" https_proxy="" all_proxy=""
    mkdir -p "$LOGDIR" "$DONEDIR" "$OUTDIR" "$INDEXDIR"
fi

# ---- STEP 0: retriever provenance fail-closed gate (BGE weight sha256) --------
echo "[#144] STEP 0: retriever provenance gate (BGE-large-en-v1.5 sha256) on CPU"
if [ "$RUN" = "1" ]; then
    $PYBIN $DRIVER --mode provenance --retriever_path $RETRIEVER >"$LOGDIR/provenance.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "[#144] PROVENANCE GATE FAILED — BGE sha mismatch; aborting."
        tail -20 "$LOGDIR/provenance.out"; exit 6
    fi
    echo "[#144] provenance PASS."
    # NOTE: the per-shard run_cell also fail-closes on the flagship LoRA sha
    # (EXPECTED_LORA_SHA) — a wrong adapter aborts that shard immediately.
else
    echo "  \$ $PYBIN $DRIVER --mode provenance --retriever_path $RETRIEVER"
fi

# ---- STEP 1: build the (family|task|length|shard) job queue ------------------
echo "[#144] STEP 1: building job queue ($NUM_SHARDS shards/cell)"
NJOBS=0
[ "$RUN" = "1" ] && : > "$QUEUE"
_emit() {  # family task length
    local fam="$1" task="$2" len="$3" sh
    for sh in $(seq 0 $((NUM_SHARDS - 1))); do
        [ "$RUN" = "1" ] && echo "$fam|$task|$len|$sh" >> "$QUEUE"
        NJOBS=$((NJOBS + 1))
    done
}
# longest lengths first (LPT: heavy cells drain into the pool early)
for len in $RULER_LENGTHS; do for t in $RULER_TASKS; do _emit ruler "$t" "$len"; done; done
for len in $BABI_LENGTHS;  do for t in $BABI_TASKS;  do _emit babilong "$t" "$len"; done; done
[ "$DO_LOCOMO" = "1" ] && _emit locomo NONE all
echo "[#144] queued $NJOBS shard-jobs (tasks x lengths x $NUM_SHARDS shards)."

echo "[#144] example per-shard command:"
echo "    CUDA_VISIBLE_DEVICES=<g> $PYBIN $DRIVER --mode run $COMMON \\"
echo "        --family babilong --task qa5 --length 8k \\"
echo "        --num_shards $NUM_SHARDS --shard_index 0 --device cuda:0"

# ---- STEP 2: 8-GPU flock task pool -------------------------------------------
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
            IFS='|' read -r fam task len sh <<< "$job"
            local targ=""; [ "$task" != "NONE" ] && targ="--task $task"
            local tag="${fam}_${task}_${len}_shard${sh}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            echo "[gpu$gpu] START $job"
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER --mode run $COMMON \
                --family "$fam" $targ --length "$len" \
                --num_shards "$NUM_SHARDS" --shard_index "$sh" --device cuda:0 \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[#144] STEP 2: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[#144] all workers done."

    # ---- STEP 3: aggregate (all-tasks fail-closed guard, exit 5 if missing) --
    echo "[#144] STEP 3: aggregate + all-tasks guard"
    $PYBIN $DRIVER --mode aggregate --output_dir "$OUTDIR" \
        --require_family $REQUIRE >"$LOGDIR/aggregate.out" 2>&1
    AGG_RC=$?
    cat "$LOGDIR/aggregate.out"
    [ $AGG_RC -ne 0 ] && { echo "[#144] AGGREGATE GUARD FAILED rc=$AGG_RC"; exit $AGG_RC; }
    echo "[#144] COMPLETE. shard jsonl -> $OUTDIR/  aggregate -> $OUTDIR/aggregate.json"
else
    echo "[#144] STEP 2/3 (workers + aggregate) — DRY: not launched."
    echo "[#144] aggregate (CPU) command MAIN runs after all shards finish:"
    echo "    $PYBIN $DRIVER --mode aggregate --output_dir $OUTDIR --require_family $REQUIRE"
    echo "[#144] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
