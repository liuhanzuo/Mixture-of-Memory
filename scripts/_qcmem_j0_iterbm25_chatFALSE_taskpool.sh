#!/usr/bin/env bash
# ============================================================================
# CoMem P0#1 DECISION CONTROL — split depth j=0, iter_bm25 retrieval,
# chat=False, 8-GPU task-pool dynamic scheduler (2026-07-25, task #71).
#
# PURPOSE — answer the reviewer concern "is CoMem's win just retrieval?"
#   j=0 means NO depth-axis compression: we still retrieve the top-k chunks
#   with the SAME iter_bm25 selector / budget as the flagship, but resume at
#   layer 0 -> the retrieved pack is recomputed through ALL 36 layers (a full
#   re-encode of only the selected chunks). This is the "retrieval + full
#   recompute, no depth split, no training" baseline. It anchors two clean
#   single-variable comparisons:
#     (a) vs #65 adapter-free j=9 (no-lora): ONLY var = split depth (0 vs 9)
#         -> isolates the depth-axis partition's contribution beyond retrieval.
#     (b) vs KV-Direct (full context, NO retrieval): both full recompute
#         -> isolates the retrieval contribution ("is the win just retrieval").
#   The distillation teacher itself uses split j=0, so j=0 is a valid
#   degenerate CoMem case.
#
# FIXED config (single variable vs flagship = resume_j and NO lora):
#   resume_j 0 | NO --lora_adapter | selector iter_bm25 (hop topk 4) | topk 12
#   | sink bos | chunk_size 512 | NO --use_chat_template (chat=False)
#   model models/Qwen3-8b-local
#
# Coverage (the two benchmarks where CoMem BEATS the full-ctx upper bound,
# i.e. where the "just retrieval?" question actually bites; BABILong qa1/qa2
# come essentially free at <=32k and keep the row cell-comparable):
#   BABILong : {qa1,qa2,qa5} x {0k,1k,2k,4k,8k,16k,32k}  limit=100  4 shards
#   LoCoMo   : full (max_samples=-1)  8 shards  (local F1 now; GPT-4o judge later)
#
# Scheduler: identical flock task-pool to #65 (per-job .done -> resumable).
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }

PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_cache/datasets}"
export http_proxy="" https_proxy="" all_proxy=""   # eval is fully offline

MODEL="${MODEL:-models/Qwen3-8b-local}"
GPUS=(0 1 2 3 4 5 6 7)

# ---- FIXED j=0 control config (single var vs flagship = resume_j + no lora) ----
RESUME_J=0
SELECTOR=iter_bm25
TOPK=12
ITER_HOP_TOPK=4
SINK=bos
CHUNK=512
COMMON="--resume_j $RESUME_J --selector $SELECTOR --topk $TOPK --iter_hop_topk $ITER_HOP_TOPK --sink_tokens $SINK --chunk_size $CHUNK"
# NOTE: no --lora_adapter (=> adapter-free/j0) and no --use_chat_template (=> chat=False)

LOGDIR="logs/qcmem_j0_iterbm25_chatFALSE"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

BL_DIR="babilong_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE"
LC_DIR="locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE"

# ---------------------------------------------------------------------------
# Build the job pool. Line format (delimiter '|'):
#   BENCH | ATASK | ALEN | SHARD | NSHARD | OUTNAME
# ---------------------------------------------------------------------------
: > "$QUEUE"
# (1) LoCoMo (full doc pack, heaviest -> emit first)
for si in 0 1 2 3 4 5 6 7; do
  echo "locomo|-|-|$si|8|-" >> "$QUEUE"
done
# (2) BABILong (short, <=32k)
for L in 0k 1k 2k 4k 8k 16k 32k; do
  for si in 0 1 2 3; do
    echo "babilong|qa1 qa2 qa5|$L|$si|4|qcmem_8b_zeroshot_j0_iterbm25_chatFALSE_$L" >> "$QUEUE"
  done
done
NTASKS=$(wc -l < "$QUEUE")
echo "[$(date)] j0 control task-pool built: $NTASKS jobs | model=$MODEL | resume_j=$RESUME_J selector=$SELECTOR topk=$TOPK hop=$ITER_HOP_TOPK sink=$SINK chunk=$CHUNK | NO lora, NO chat_template | 8 GPU workers"

# ---- atomic pop one job line (flock) ----
pop_job() {
  exec 9>"$LOCK"; flock 9
  local line; line="$(head -n 1 "$QUEUE")"
  if [ -n "$line" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
  flock -u 9; exec 9>&-
  echo "$line"
}

# ---- run one job on GPU $g ----
run_job() {
  local g="$1" bench="$2" atask="$3" alen="$4" shard="$5" nshard="$6" outname="$7"
  local jobid; jobid="$(echo "${bench}_${atask}_${alen}_s${shard}of${nshard}_${outname}" | tr ' /' '__')"
  local marker="$DONEDIR/$jobid.done"
  local log="$LOGDIR/$jobid.log"
  if [ -f "$marker" ]; then echo "[skip] GPU$g $jobid (done)"; return 0; fi
  echo "[$(date)] GPU$g START $jobid"
  local rc=0
  case "$bench" in
    locomo)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_locomo.py \
        --model_path "$MODEL" $COMMON \
        --locomo_data locomo/data/locomo10.json --max_samples -1 \
        --output_dir "$LC_DIR" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    babilong)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_babilong.py \
        --model_path "$MODEL" $COMMON \
        --tasks $atask --lengths $alen --limit 100 \
        --results_folder "$BL_DIR" --output_name "$outname" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 --max_new_tokens 20 \
        --use_instruction --use_examples --use_post_prompt \
        </dev/null >"$log" 2>&1; rc=$? ;;
    *) echo "[$(date)] GPU$g UNKNOWN bench=$bench"; return 1 ;;
  esac
  if [ "$rc" -eq 0 ]; then touch "$marker"; echo "[$(date)] GPU$g DONE  $jobid";
  else echo "[$(date)] GPU$g FAIL  $jobid rc=$rc (see $log)"; fi
}

# ---- a GPU worker: pop jobs until queue empty ----
gpu_worker() {
  local g="$1"
  sleep $((g * 10))   # stagger cold model loads
  while true; do
    local line; line="$(pop_job)"
    [ -z "$line" ] && break
    IFS='|' read -r BENCH ATASK ALEN SHARD NSHARD OUTNAME <<< "$line"
    run_job "$g" "$BENCH" "$ATASK" "$ALEN" "$SHARD" "$NSHARD" "$OUTNAME"
  done
  echo "[$(date)] GPU$g drained — queue empty"
}

PIDS=()
for g in "${GPUS[@]}"; do gpu_worker "$g" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done

# ---------------------------------------------------------------------------
# Scoring (official scorers only).
# ---------------------------------------------------------------------------
SUMMARY="$LOGDIR/SUMMARY.txt"
{
  echo "======= CoMem P0#1 j=0 control (iter_bm25, chat=False) SUMMARY ======="
  echo "date: $(date)"
  echo "config: resume_j=$RESUME_J NO-lora selector=$SELECTOR topk=$TOPK hop=$ITER_HOP_TOPK sink=$SINK chunk=$CHUNK chat=False"
  echo
  echo "########## BABILong (compare_answers) ##########"
  $PYBIN scripts/score_nested_babilong.py "$BL_DIR" --expect -1 2>&1
  echo
  echo "########## LoCoMo (F1; GPT-4o judge run separately) ##########"
  $PYBIN scripts/eval_qcmem_locomo.py --score_only --output_dir "$LC_DIR" 2>&1
  echo
  echo "======= END SUMMARY $(date) ======="
} | tee "$SUMMARY"

touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] SCHED_DONE — summary at $SUMMARY"
