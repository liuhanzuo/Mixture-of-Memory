#!/usr/bin/env bash
# ============================================================================
# CacheBlend baseline (Paper A #143 / A-P1.3) — 8-GPU flock task pool.
#
# CacheBlend (Yao et al., EuroSys'25, arXiv:2405.16444) = full-depth per-chunk KV
# reuse + global RoPE reindex + selective boundary-token recompute (knob r).
# SINGLE-VARIABLE control vs flagship CoMem: SAME selector (iter_bm25, hop=4),
# SAME chunk=512 / topk=12 / sink=bos / pack order — the ONLY difference is the
# cache object (full 36-layer KV, 144 KiB/tok, vs one depth-12 residual h_j,
# 8 KiB/tok). r sweeps {0.0 (pure reuse floor), 0.10, 0.15, 0.18}; r=1.0 is the
# full-prefill upper bound / self-test gate (not swept here).
#
# Flagship eval config (unified with all Paper A): Qwen3-8B models/Qwen3-8b-local,
# selector=iter_bm25 topk=12 iter_hop_topk=4 iter_rounds=0, chunk_size=512,
# sink=bos, chat_template=False (BASE LM — model has NO SFT/RL so a chat template
# is unfair; do NOT pass --use_chat_template), enable_thinking=False, bf16, sdpa,
# seed=42, n=100/cell. Results -> bench_results/cacheblend/.
#
# Eval targets:
#   RULER  : {niah_single_2, niah_multikey_1, variable_tracking} x {4k,8k,16k,32k}
#            (niah / mk / vt, cohort A + B)  -> eval_ruler_qcmem.py
#   LoCoMo : full (1986 QA)                  -> eval_qcmem_locomo.py
#   BABILong: qa5 x {0k,1k,2k,4k,8k,16k}     -> eval_qcmem_babilong.py
#
# 8-GPU flock task pool (same pattern as _run_p0_19_ruler_paired.sh): one shared
# queue of (bench|task|length|r|shard) jobs; whichever GPU is free pops the next
# job -> dynamic load balancing (long 32k / high-r cells never starve a GPU).
#
# Usage (MAIN, on a FREE diskB torch-base node — .104 / .82 / .73):
#   PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_cacheblend_8gpu.sh \
#     >logs/cacheblend/sched.out 2>&1 &
#   # dry preview (default): omit RUN=1 — prints commands + job count, no forward.
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
RUN="${RUN:-0}"                           # 0 = DRY (print only) ; 1 = execute

MODEL="${MODEL:-models/Qwen3-8b-local}"
SELECTOR="${SELECTOR:-iter_bm25}"         # matched to flagship CoMem (single-variable)
TOPK="${TOPK:-12}"
HOP="${HOP:-4}"
CHUNK="${CHUNK:-512}"
LIMIT="${LIMIT:-100}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SEED="${SEED:-42}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"

# recompute-ratio sweep (the ONLY CacheBlend knob). r=1.0 excluded here (it is the
# self-test gate / full-prefill upper bound; add "1.0" to RS to also eval the ceil).
RS="${RS:-0.0 0.10 0.15 0.18}"

# per-benchmark task/length grids (override to shrink a preview run).
RULER_TASKS="${RULER_TASKS:-niah_single_2 niah_multikey_1 variable_tracking}"
RULER_LENGTHS="${RULER_LENGTHS:-32k 16k 8k 4k}"   # longest first (LPT: heavy cells drain early)
BABI_TASKS="${BABI_TASKS:-qa5}"
BABI_LENGTHS="${BABI_LENGTHS:-16k 8k 4k 2k 1k 0k}"
DO_LOCOMO="${DO_LOCOMO:-1}"

RESULTS="${RESULTS:-bench_results/cacheblend}"
LOGDIR="${LOGDIR:-logs/cacheblend}"

RULER_DRIVER="scripts/eval_ruler_qcmem.py"
BABI_DRIVER="scripts/eval_qcmem_babilong.py"
LOCOMO_DRIVER="scripts/eval_qcmem_locomo.py"
LOCOMO_DATA="${LOCOMO_DATA:-locomo/data/locomo10.json}"

# r=0.18 -> "r018" tag (stable, filesystem-safe).
rtag() { echo "r$(printf '%s' "$1" | tr -d '.' )"; }

echo "============================================================"
echo "[cacheblend] PROJECT_ROOT=$PROJECT_ROOT  PYBIN=$PYBIN  RUN=$RUN"
echo "[cacheblend] model=$MODEL selector=$SELECTOR topk=$TOPK hop=$HOP chunk=$CHUNK"
echo "[cacheblend] limit=$LIMIT shards=$NUM_SHARDS seed=$SEED  chat_template=False"
echo "[cacheblend] recompute-ratio sweep RS=[$RS]  (r=1.0 = self-test/full-prefill ceil)"
echo "[cacheblend] RULER tasks=[$RULER_TASKS] lengths=[$RULER_LENGTHS]"
echo "[cacheblend] BABILong tasks=[$BABI_TASKS] lengths=[$BABI_LENGTHS]  LoCoMo=$DO_LOCOMO"
echo "[cacheblend] RESULTS=$RESULTS  LOGDIR=$LOGDIR"
[ "$RUN" != "1" ] && echo "[cacheblend] *** DRY-RUN (RUN!=1): commands only, no forward ***"
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
    mkdir -p "$LOGDIR" "$LOGDIR/done" "$RESULTS"
fi

QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"
DONEDIR="$LOGDIR/done"

# ---- STEP 0: fail-closed correctness gate — CacheBlend self-test -------------
# (A) RoPE reindex exactness, (B) r=1.0 == vanilla full prefill token-by-token,
# (C) r=0.0 finite. Forces fp32; one GPU; abort the whole run on failure.
echo "[cacheblend] STEP 0: CacheBlend self-test gate (reindex + r=1.0==full prefill, fp32)"
SELFTEST_CMD="$PYBIN $BABI_DRIVER --model_path $MODEL --baseline cacheblend \
--self_test --output_name _cb_selftest --results_folder $RESULTS \
--tasks qa5 --lengths 1k --chunk_size $CHUNK"
if [ "$RUN" = "1" ]; then
    CUDA_VISIBLE_DEVICES=0 $SELFTEST_CMD >"$LOGDIR/selftest.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "[cacheblend] SELF-TEST GATE FAILED — aborting."; tail -30 "$LOGDIR/selftest.out"; exit 6
    fi
    echo "[cacheblend] self-test PASS."
else
    echo "  \$ CUDA_VISIBLE_DEVICES=0 $SELFTEST_CMD"
fi

# ---- STEP 1: build the (bench|task|length|r|shard) job queue -----------------
echo "[cacheblend] STEP 1: building job queue"
NJOBS=0
[ "$RUN" = "1" ] && : > "$QUEUE"
_emit() {  # bench task length r
    local bench="$1" task="$2" len="$3" r="$4" sh
    for sh in $(seq 0 $((NUM_SHARDS - 1))); do
        [ "$RUN" = "1" ] && echo "$bench|$task|$len|$r|$sh" >> "$QUEUE"
        NJOBS=$((NJOBS + 1))
    done
}
for r in $RS; do
    for t in $RULER_TASKS; do
        for len in $RULER_LENGTHS; do _emit ruler "$t" "$len" "$r"; done
    done
    for t in $BABI_TASKS; do
        for len in $BABI_LENGTHS; do _emit babilong "$t" "$len" "$r"; done
    done
    [ "$DO_LOCOMO" = "1" ] && _emit locomo full NA "$r"
done
echo "[cacheblend] queued $NJOBS shard-jobs (RS x tasks x lengths x $NUM_SHARDS shards)."

echo "[cacheblend] example per-shard commands:"
echo "  RULER:    CUDA_VISIBLE_DEVICES=<g> $PYBIN $RULER_DRIVER --model_path $MODEL \\"
echo "              --baseline cacheblend --recompute_ratio <r> --selector $SELECTOR --topk $TOPK \\"
echo "              --iter_hop_topk $HOP --iter_rounds 0 --chunk_size $CHUNK --sink_tokens bos \\"
echo "              --dtype bfloat16 --attn_impl sdpa --seed $SEED --limit $LIMIT \\"
echo "              --ruler_tasks <task> --lengths <len> --num_shards $NUM_SHARDS --shard_index <s> \\"
echo "              --device cuda:0 --results_folder $RESULTS --output_name cb_ruler_<task>_<rtag>"
echo "  BABILong: CUDA_VISIBLE_DEVICES=<g> $PYBIN $BABI_DRIVER --model_path $MODEL \\"
echo "              --baseline cacheblend --recompute_ratio <r> --selector $SELECTOR --topk $TOPK \\"
echo "              --iter_hop_topk $HOP --iter_rounds 0 --chunk_size $CHUNK --sink_tokens bos \\"
echo "              --dtype bfloat16 --attn_impl sdpa --seed $SEED --limit $LIMIT \\"
echo "              --tasks <task> --lengths <len> --num_shards $NUM_SHARDS --shard_index <s> \\"
echo "              --device cuda:0 --results_folder $RESULTS --output_name cb_babilong_<task>_<rtag>"
echo "  LoCoMo:   CUDA_VISIBLE_DEVICES=<g> $PYBIN $LOCOMO_DRIVER --model_path $MODEL \\"
echo "              --baseline cacheblend --recompute_ratio <r> --selector $SELECTOR --topk $TOPK \\"
echo "              --iter_hop_topk $HOP --iter_rounds 0 --chunk_size $CHUNK --sink_tokens bos \\"
echo "              --dtype bfloat16 --attn_impl sdpa --limit $LIMIT \\"
echo "              --locomo_data $LOCOMO_DATA --num_shards $NUM_SHARDS --shard_index <s> \\"
echo "              --device cuda:0 --output_dir $RESULTS/cb_locomo_<rtag>"

# ---- STEP 2: 8-GPU flock task pool -------------------------------------------
if [ "$RUN" = "1" ]; then
    read -r -a GPUS <<< "$GPUS_STR"
    pop_job() {
        ( flock -x 200
          local first; first=$(head -n 1 "$QUEUE")
          if [ -n "$first" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
          echo "$first" ) 200>"$LOCK"
    }
    run_job() {  # gpu bench task len r sh
        local gpu="$1" bench="$2" task="$3" len="$4" r="$5" sh="$6"
        local rt; rt=$(rtag "$r")
        case "$bench" in
            ruler)
                CUDA_VISIBLE_DEVICES=$gpu $PYBIN $RULER_DRIVER --model_path "$MODEL" \
                    --baseline cacheblend --recompute_ratio "$r" \
                    --selector "$SELECTOR" --topk "$TOPK" --iter_hop_topk "$HOP" \
                    --iter_rounds 0 --chunk_size "$CHUNK" --sink_tokens bos \
                    --dtype bfloat16 --attn_impl sdpa --seed "$SEED" --limit "$LIMIT" \
                    --ruler_tasks "$task" --lengths "$len" \
                    --num_shards "$NUM_SHARDS" --shard_index "$sh" --device cuda:0 \
                    --results_folder "$RESULTS" --output_name "cb_ruler_${task}_${rt}"
                ;;
            babilong)
                CUDA_VISIBLE_DEVICES=$gpu $PYBIN $BABI_DRIVER --model_path "$MODEL" \
                    --baseline cacheblend --recompute_ratio "$r" \
                    --selector "$SELECTOR" --topk "$TOPK" --iter_hop_topk "$HOP" \
                    --iter_rounds 0 --chunk_size "$CHUNK" --sink_tokens bos \
                    --dtype bfloat16 --attn_impl sdpa --seed "$SEED" --limit "$LIMIT" \
                    --tasks "$task" --lengths "$len" \
                    --num_shards "$NUM_SHARDS" --shard_index "$sh" --device cuda:0 \
                    --results_folder "$RESULTS" --output_name "cb_babilong_${task}_${rt}"
                ;;
            locomo)
                CUDA_VISIBLE_DEVICES=$gpu $PYBIN $LOCOMO_DRIVER --model_path "$MODEL" \
                    --baseline cacheblend --recompute_ratio "$r" \
                    --selector "$SELECTOR" --topk "$TOPK" --iter_hop_topk "$HOP" \
                    --iter_rounds 0 --chunk_size "$CHUNK" --sink_tokens bos \
                    --dtype bfloat16 --attn_impl sdpa --limit "$LIMIT" \
                    --locomo_data "$LOCOMO_DATA" \
                    --num_shards "$NUM_SHARDS" --shard_index "$sh" --device cuda:0 \
                    --output_dir "$RESULTS/cb_locomo_${rt}"
                ;;
        esac
    }
    worker() {
        local gpu="$1"
        while true; do
            local job; job=$(pop_job); [ -z "$job" ] && break
            IFS='|' read -r bench task len r sh <<< "$job"
            local tag="${bench}_${task}_${len}_$(rtag "$r")_shard${sh}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            echo "[gpu$gpu] START $job"
            run_job "$gpu" "$bench" "$task" "$len" "$r" "$sh" \
                >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[cacheblend] STEP 2: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[cacheblend] all workers done. shard CSVs/JSON -> $RESULTS/"
    echo "[cacheblend] NEXT (MAIN): merge shards (score_nested_babilong.py for RULER/BABILong;"
    echo "[cacheblend]              --score_only for LoCoMo) and read cacheblend_kv_bytes_per_tok"
    echo "[cacheblend]              / prefill_latency_ms / peak_mem / recompute_ratio per cell."
else
    echo "[cacheblend] STEP 2 — DRY: not launched. Re-run with RUN=1 on a FREE diskB node."
fi
