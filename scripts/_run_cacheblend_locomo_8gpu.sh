#!/usr/bin/env bash
# ============================================================================
# CacheBlend baseline (Paper A #143 / A-P1.3) — LoCoMo-ONLY 8-GPU flock pool.
#
# WHY A DEDICATED LoCoMo LAUNCHER (not the shared scripts/_run_cacheblend_8gpu.sh):
#   The shared cacheblend scheduler passes `--limit $LIMIT` (=100) to EVERY
#   driver. That is correct for RULER/BABILong (both define --limit), but the
#   LoCoMo driver (scripts/eval_qcmem_locomo.py) has NO --limit flag — it caps
#   samples with --max_samples (default -1 = all 1986 QA). So the shared
#   scheduler's LoCoMo path would (a) die on `unrecognized arguments: --limit`
#   and (b) even if renamed, wrongly cap LoCoMo to 100 of its 1986 QA. LoCoMo
#   was deferred when the shared scheduler was authored, so that path was never
#   exercised. The shared scheduler is running LIVE on .73 and MUST NOT be
#   edited — hence this standalone launcher with the correct LoCoMo args.
#
# SINGLE-VARIABLE control vs flagship CoMem (identical to the RULER/BABILong
# cacheblend cells): SAME selector (iter_bm25, hop=4), SAME chunk=512 / topk=12
# / sink=bos / pack order. CacheBlend is training-free (full 36-layer per-chunk
# KV reused via global-RoPE reindex + selective boundary recompute, knob r) so
# the driver AUTO-DROPS any LoRA (eval_qcmem_locomo.py:869-876) — do NOT pass
# --lora_adapter; the only difference vs CoMem is the cache object. r sweeps
# {0.0, 0.10, 0.15, 0.18} (r=1.0 is the self-test/full-prefill ceil, gated in
# STEP 0, not swept).
#
# Protocol (unified with all Paper A #143): Qwen3-8B models/Qwen3-8b-local,
# selector=iter_bm25 topk=12 iter_hop_topk=4 iter_rounds=0, chunk_size=512,
# sink=bos, chat_template=False (BASE LM — do NOT pass --use_chat_template),
# enable_thinking=False, bf16, sdpa. LoCoMo = FULL 1986 QA (--max_samples -1).
# Results -> bench_results/cacheblend/cb_locomo_<rtag>/ (disjoint from the .73
# run's cb_ruler_* / cb_babilong_* files — no write collision on the shared FS).
#
# 8-GPU flock task pool: shared queue of (r|shard) jobs; whichever GPU is free
# pops the next job (LPT: r=0.0 pure-reuse cheapest, high-r heaviest — but every
# r has the same 8 shards so the pool self-balances). .done markers for
# idempotency (re-run skips completed shards). After the pool drains, an offline
# F1/EM --score_only merge is run per r-dir (LLM-judge grading is a post-hoc
# pass MAIN can add later over the same preds JSONL).
#
# Usage (on a FREE diskB torch-base node — here .82, zwfy6 shared FS with .73):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   RUN=1 setsid nohup bash scripts/_run_cacheblend_locomo_8gpu.sh \
#     >logs/cacheblend_locomo82/sched.out 2>&1 &
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
MAX_SAMPLES="${MAX_SAMPLES:--1}"          # -1 = ALL 1986 QA (LoCoMo full)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"    # locomo driver default (short answers)
NUM_SHARDS="${NUM_SHARDS:-8}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"

# recompute-ratio sweep (the ONLY CacheBlend knob). Matches the .73 #143 sweep.
RS="${RS:-0.0 0.10 0.15 0.18}"

RESULTS="${RESULTS:-bench_results/cacheblend}"
LOGDIR="${LOGDIR:-logs/cacheblend_locomo82}"   # disjoint from the .73 run's logs/cacheblend

LOCOMO_DRIVER="scripts/eval_qcmem_locomo.py"
LOCOMO_DATA="${LOCOMO_DATA:-locomo/data/locomo10.json}"

# r=0.18 -> "r018" tag (stable, filesystem-safe).
rtag() { echo "r$(printf '%s' "$1" | tr -d '.' )"; }

echo "============================================================"
echo "[cb-locomo] PROJECT_ROOT=$PROJECT_ROOT  PYBIN=$PYBIN  RUN=$RUN"
echo "[cb-locomo] model=$MODEL selector=$SELECTOR topk=$TOPK hop=$HOP chunk=$CHUNK"
echo "[cb-locomo] max_samples=$MAX_SAMPLES (=-1 -> ALL 1986 QA) shards=$NUM_SHARDS chat_template=False"
echo "[cb-locomo] recompute-ratio sweep RS=[$RS]  (r=1.0 = self-test/full-prefill ceil)"
echo "[cb-locomo] RESULTS=$RESULTS  LOGDIR=$LOGDIR  data=$LOCOMO_DATA"
[ "$RUN" != "1" ] && echo "[cb-locomo] *** DRY-RUN (RUN!=1): commands only, no forward ***"
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
# Runs the LoCoMo driver's own cacheblend self-test (reindex exactness +
# r=1.0 == vanilla full prefill token-by-token, fp32). Same code path as the
# .73 babilong self-test; re-run here so this node is self-contained.
echo "[cb-locomo] STEP 0: CacheBlend self-test gate (reindex + r=1.0==full prefill, fp32)"
SELFTEST_CMD="$PYBIN $LOCOMO_DRIVER --model_path $MODEL --baseline cacheblend \
--self_test --chunk_size $CHUNK --dtype float32 --device cuda:0"
if [ "$RUN" = "1" ]; then
    CUDA_VISIBLE_DEVICES=0 $SELFTEST_CMD >"$LOGDIR/selftest.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "[cb-locomo] SELF-TEST GATE FAILED — aborting."; tail -40 "$LOGDIR/selftest.out"; exit 6
    fi
    echo "[cb-locomo] self-test PASS."
else
    echo "  \$ CUDA_VISIBLE_DEVICES=0 $SELFTEST_CMD"
fi

# ---- STEP 1: build the (r|shard) job queue -----------------------------------
echo "[cb-locomo] STEP 1: building job queue"
NJOBS=0
[ "$RUN" = "1" ] && : > "$QUEUE"
for r in $RS; do
    for sh in $(seq 0 $((NUM_SHARDS - 1))); do
        [ "$RUN" = "1" ] && echo "$r|$sh" >> "$QUEUE"
        NJOBS=$((NJOBS + 1))
    done
done
echo "[cb-locomo] queued $NJOBS shard-jobs (RS x $NUM_SHARDS shards)."
echo "[cb-locomo] per-shard command template:"
echo "  CUDA_VISIBLE_DEVICES=<g> $PYBIN $LOCOMO_DRIVER --model_path $MODEL \\"
echo "    --baseline cacheblend --recompute_ratio <r> --selector $SELECTOR --topk $TOPK \\"
echo "    --iter_hop_topk $HOP --iter_rounds 0 --chunk_size $CHUNK --sink_tokens bos \\"
echo "    --dtype bfloat16 --attn_impl sdpa --max_samples $MAX_SAMPLES --max_new_tokens $MAX_NEW_TOKENS \\"
echo "    --locomo_data $LOCOMO_DATA --num_shards $NUM_SHARDS --shard_index <s> --device cuda:0 \\"
echo "    --output_dir $RESULTS/cb_locomo_<rtag>       # NOTE: no --limit, no --lora_adapter"

# ---- STEP 2: 8-GPU flock task pool -------------------------------------------
if [ "$RUN" = "1" ]; then
    read -r -a GPUS <<< "$GPUS_STR"
    pop_job() {
        ( flock -x 200
          local first; first=$(head -n 1 "$QUEUE")
          if [ -n "$first" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
          echo "$first" ) 200>"$LOCK"
    }
    run_job() {  # gpu r sh
        local gpu="$1" r="$2" sh="$3"
        local rt; rt=$(rtag "$r")
        CUDA_VISIBLE_DEVICES=$gpu $PYBIN $LOCOMO_DRIVER --model_path "$MODEL" \
            --baseline cacheblend --recompute_ratio "$r" \
            --selector "$SELECTOR" --topk "$TOPK" --iter_hop_topk "$HOP" \
            --iter_rounds 0 --chunk_size "$CHUNK" --sink_tokens bos \
            --dtype bfloat16 --attn_impl sdpa \
            --max_samples "$MAX_SAMPLES" --max_new_tokens "$MAX_NEW_TOKENS" \
            --locomo_data "$LOCOMO_DATA" \
            --num_shards "$NUM_SHARDS" --shard_index "$sh" --device cuda:0 \
            --output_dir "$RESULTS/cb_locomo_${rt}"
    }
    worker() {
        local gpu="$1"
        while true; do
            local job; job=$(pop_job); [ -z "$job" ] && break
            IFS='|' read -r r sh <<< "$job"
            local tag="locomo_$(rtag "$r")_shard${sh}"
            [ -f "$DONEDIR/$tag.done" ] && { echo "[gpu$gpu] SKIP $tag"; continue; }
            echo "[gpu$gpu] START $job"
            run_job "$gpu" "$r" "$sh" >"$LOGDIR/$tag.out" 2>&1 && touch "$DONEDIR/$tag.done"
            echo "[gpu$gpu] END $job rc=$?"
        done
        echo "[gpu$gpu] worker exit"
    }
    echo "[cb-locomo] STEP 2: launching ${#GPUS[@]} GPU workers"
    pids=()
    for g in "${GPUS[@]}"; do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[cb-locomo] all workers done."

    # ---- STEP 3: offline F1/EM merge per r-dir (no LLM judge here) ------------
    echo "[cb-locomo] STEP 3: --score_only merge (offline F1/EM) per r-dir"
    for r in $RS; do
        rt=$(rtag "$r")
        d="$RESULTS/cb_locomo_${rt}"
        [ -d "$d" ] || { echo "[cb-locomo] MISSING $d — skip merge"; continue; }
        echo "[cb-locomo] merge $d"
        $PYBIN $LOCOMO_DRIVER --score_only --output_dir "$d" \
            >"$LOGDIR/score_${rt}.out" 2>&1 \
            && echo "[cb-locomo] scored $d -> scores.json" \
            || { echo "[cb-locomo] SCORE FAILED $d"; tail -20 "$LOGDIR/score_${rt}.out"; }
    done
    echo "[cb-locomo] DONE. preds + scores.json -> $RESULTS/cb_locomo_<rtag>/"
    echo "[cb-locomo] NEXT (MAIN): read scores.json (F1/EM); optional LLM-judge is a"
    echo "[cb-locomo]              post-hoc --use_llm_judge --score_only pass over the same preds."
else
    echo "[cb-locomo] STEP 2/3 — DRY: not launched. Re-run with RUN=1 on a FREE diskB node."
fi
