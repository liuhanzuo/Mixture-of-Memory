#!/usr/bin/env bash
# ============================================================================
# P0.19 RULER leg — retrieval-recall x in-pack-readout decomposition, PAIRED.
#
# Closes the RULER leg of P0.19 (task #135). The CoMem(j=12) vs j=0(RAG upper
# bound) accuracy gap is split into (a) SELECTOR MISS (gold support chunk not in
# the iter_bm25 top-k pack, answer-independent recall) and (b) CACHED-STATE
# READOUT FAILURE (gold in pack but the mid-depth resume can't read it out).
#
# THREE arms, byte-identically paired (same crc32 seed => same sample per index),
# mirroring the LongEval P0.19 table:
#   j0        : resume_j=0, NO LoRA          -> RAG full-depth recompute upper bound
#   j12frozen : resume_j=12, NO LoRA         -> readout gap (adapter frozen)
#   j12lora   : resume_j=12, flagship LoRA   -> CoMem RULER leg (learnable recovery)
#
# Flagship config (unified with all Paper A): Qwen3-8B models/Qwen3-8b-local,
# selector=iter_bm25 topk=12 iter_hop_topk=4, chunk_size=512, sink=bos,
# chat_template=False (base LM, no BOS beyond tokenizer default), bf16, sdpa,
# seed=42. Cohort = niah_multikey_1 x {8k,16k} n=100 (== Phase B's RULER cell).
#
# Seed-pairing is guaranteed two ways: (1) eval_ruler_qcmem seeds the sample RNG
# with zlib.crc32 (PYTHONHASHSEED-independent, committed d1e1389); (2) each arm
# emits per-sample input_ids_sha256 and the CPU analyzer FAIL-CLOSES if any two
# arms (or the regenerated prompt) disagree — no unpaired numbers are ever emitted.
# We ALSO pin PYTHONHASHSEED=0 everywhere for belt-and-braces determinism.
#
# 8-GPU flock task-pool (same pattern as _run_p1_9_dense_rag_8gpu.sh /
# _run_p0_20_phaseB_dense.sh): 3 arms x 2 lengths x 8 shards = 48 shard-jobs.
#
# Usage (MAIN, on a FREE diskB H20 node — .104):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_p0_19_ruler_paired.sh \
#     >/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/p0_19_ruler/sched.out 2>&1 &
#   # dry preview (default): omit RUN=1 — prints commands only, no forward.
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYBIN="${PYTHON_BIN:-.venv/bin/python}"
RUN="${RUN:-0}"                           # 0 = DRY (print only) ; 1 = execute

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
RESUME_J="${RESUME_J:-12}"
SELECTOR="${SELECTOR:-iter_bm25}"
TOPK="${TOPK:-12}"
HOP="${HOP:-4}"
CHUNK="${CHUNK:-512}"
LIMIT="${LIMIT:-100}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SEED="${SEED:-42}"
RULER_TASKS="${RULER_TASKS:-niah_multikey_1}"
RULER_LENGTHS="${RULER_LENGTHS:-16k 8k}"   # longest first (LPT: heavy cells drain early)
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"

RESULTS="${RESULTS:-ruler_results/p0_19_ruler_paired}"
LOGDIR="${LOGDIR:-logs/p0_19_ruler}"
OUTJSON="${OUTJSON:-paperA/p0_19_ruler_decomp.json}"

J0_NAME="p0_19_ruler_j0_iterbm25_mk_chatFALSE"
J12F_NAME="p0_19_ruler_j12frozen_iterbm25_mk_chatFALSE"
J12L_NAME="p0_19_ruler_j12lora_iterbm25_mk_chatFALSE"

DRIVER="scripts/eval_ruler_qcmem.py"
ANALYZER="scripts/analyze_p019_recall_readout.py"

# per-arm QCMem config (chat_template OFF = default; do NOT pass --use_chat_template)
COMMON="--model_path $MODEL --selector $SELECTOR --topk $TOPK \
--iter_hop_topk $HOP --iter_rounds 0 --chunk_size $CHUNK --sink_tokens bos \
--dtype bfloat16 --attn_impl sdpa --seed $SEED --limit $LIMIT \
--ruler_tasks $RULER_TASKS --lengths $RULER_LENGTHS \
--num_shards $NUM_SHARDS --results_folder $RESULTS"

echo "============================================================"
echo "[p0.19-ruler] PROJECT_ROOT=$PROJECT_ROOT  PYBIN=$PYBIN  RUN=$RUN"
echo "[p0.19-ruler] model=$MODEL lora=$LORA resume_j=$RESUME_J"
echo "[p0.19-ruler] selector=$SELECTOR topk=$TOPK hop=$HOP chunk=$CHUNK limit=$LIMIT shards=$NUM_SHARDS seed=$SEED"
echo "[p0.19-ruler] tasks=[$RULER_TASKS] lengths=[$RULER_LENGTHS]  chat_template=False"
echo "[p0.19-ruler] arms: j0(RAG,noLoRA) / j12frozen(noLoRA) / j12lora(flagship CoMem)"
echo "[p0.19-ruler] RESULTS=$RESULTS  LOGDIR=$LOGDIR  OUTJSON=$OUTJSON"
[ "$RUN" != "1" ] && echo "[p0.19-ruler] *** DRY-RUN (RUN!=1): commands only, no forward ***"
echo "============================================================"

if [ "$RUN" = "1" ]; then
    cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }
    export PYTHONUNBUFFERED=1
    export PYTHONHASHSEED=0
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
    export WANDB_MODE=offline
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$PROJECT_ROOT/.hf_cache" HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
    export http_proxy="" https_proxy="" all_proxy=""
    mkdir -p "$LOGDIR" "$LOGDIR/done" "$RESULTS" "$(dirname "$OUTJSON")"
fi

QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"
DONEDIR="$LOGDIR/done"

# ---- STEP 0: manifest/sanity fail-closed gate — QCMem j=0 self-test ----------
# (read == full forward, fp32 max|logit diff| < 1e-4). One GPU; abort on failure.
echo "[p0.19-ruler] STEP 0: QCMem self-test gate (j=0 read == full forward, fp32)"
if [ "$RUN" = "1" ]; then
    CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --model_path "$MODEL" --resume_j 0 \
        --self_test --chunk_size "$CHUNK" --output_name _selftest \
        --results_folder "$RESULTS" >"$LOGDIR/selftest.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "[p0.19-ruler] SELF-TEST GATE FAILED — aborting."; tail -25 "$LOGDIR/selftest.out"; exit 6
    fi
    echo "[p0.19-ruler] self-test PASS."
else
    echo "  \$ CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --model_path $MODEL --resume_j 0 --self_test --chunk_size $CHUNK --output_name _selftest --results_folder $RESULTS"
fi

# ---- STEP 1: build the (arm,length,shard) job queue --------------------------
echo "[p0.19-ruler] STEP 1: job queue (3 arms x [$RULER_LENGTHS] x $NUM_SHARDS shards)"
NJOBS=0
[ "$RUN" = "1" ] && : > "$QUEUE"
_emit() {  # arm length
    local arm="$1" len="$2" sh
    for sh in $(seq 0 $((NUM_SHARDS - 1))); do
        echo "    job: $arm|$len|$sh"
        [ "$RUN" = "1" ] && echo "$arm|$len|$sh" >> "$QUEUE"
        NJOBS=$((NJOBS + 1))
    done
}
for len in $RULER_LENGTHS; do
    _emit j0 "$len"; _emit j12frozen "$len"; _emit j12lora "$len"
done
echo "[p0.19-ruler] queued $NJOBS jobs."

echo "[p0.19-ruler] example per-shard command (j12lora):"
echo "    CUDA_VISIBLE_DEVICES=<g> $PYBIN $DRIVER $COMMON --resume_j $RESUME_J \\"
echo "        --lora_adapter $LORA --output_name $J12L_NAME --shard_index <s> --device cuda:0"

# ---- STEP 2: 8-GPU flock task pool -------------------------------------------
if [ "$RUN" = "1" ]; then
    read -r -a GPUS <<< "$GPUS_STR"
    pop_job() {
        ( flock -x 200
          local first; first=$(head -n 1 "$QUEUE")
          if [ -n "$first" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
          echo "$first" ) 200>"$LOCK"
    }
    arm_args() {  # arm -> the arm-specific driver flags
        case "$1" in
            j0)        echo "--resume_j 0 --output_name $J0_NAME" ;;
            j12frozen) echo "--resume_j $RESUME_J --output_name $J12F_NAME" ;;
            j12lora)   echo "--resume_j $RESUME_J --lora_adapter $LORA --output_name $J12L_NAME" ;;
        esac
    }
    worker() {
        local gpu="$1"
        while true; do
            local job; job=$(pop_job); [ -z "$job" ] && break
            IFS='|' read -r arm len sh <<< "$job"
            local tag="${arm}_${len}_shard${sh}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            echo "[gpu$gpu] START $job"
            CUDA_VISIBLE_DEVICES=$gpu $PYBIN $DRIVER $COMMON $(arm_args "$arm") \
                --shard_index "$sh" --device cuda:0 \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[p0.19-ruler] STEP 2: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[p0.19-ruler] all workers done."

    # ---- STEP 3: CPU analyzer join (recall x readout, fail-closed sha gate) ----
    echo "[p0.19-ruler] STEP 3: paired decomposition join (fail-closed sha256 gate)"
    $PYBIN $ANALYZER --task ruler --model_path "$MODEL" \
        --j0_dir "$RESULTS/$J0_NAME" \
        --j12_dir "$RESULTS/$J12F_NAME" \
        --flagship_dir "$RESULTS/$J12L_NAME" \
        --ruler_tasks $RULER_TASKS --lengths $RULER_LENGTHS \
        --ruler_selector "$SELECTOR" --topk "$TOPK" --iter_hop_topk "$HOP" \
        --iter_rounds 0 --chunk_size "$CHUNK" --limit "$LIMIT" --seed "$SEED" \
        --out "$OUTJSON" >"$LOGDIR/analyze.out" 2>&1
    AGG_RC=$?
    cat "$LOGDIR/analyze.out"
    [ $AGG_RC -ne 0 ] && { echo "[p0.19-ruler] ANALYZER (seed-pairing gate?) FAILED rc=$AGG_RC"; exit $AGG_RC; }
    echo "[p0.19-ruler] COMPLETE -> $OUTJSON"
else
    echo "[p0.19-ruler] STEP 2/3 — DRY: not launched."
    echo "[p0.19-ruler] analyzer join MAIN runs after shards finish:"
    echo "    $PYBIN $ANALYZER --task ruler --model_path $MODEL \\"
    echo "        --j0_dir $RESULTS/$J0_NAME --j12_dir $RESULTS/$J12F_NAME --flagship_dir $RESULTS/$J12L_NAME \\"
    echo "        --ruler_tasks $RULER_TASKS --lengths $RULER_LENGTHS --ruler_selector $SELECTOR \\"
    echo "        --topk $TOPK --iter_hop_topk $HOP --iter_rounds 0 --chunk_size $CHUNK --limit $LIMIT --seed $SEED --out $OUTJSON"
    echo "[p0.19-ruler] to execute for real on a FREE diskB node: re-run with RUN=1."
fi
