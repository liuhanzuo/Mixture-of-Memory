#!/usr/bin/env bash
# ============================================================================
# CoMem ADAPTER-FREE DEPTH SWEEP — frozen backbone (NO LoRA), split depth j in
# {6,12}, iter_bm25 retrieval, chat=False, 8-GPU task-pool scheduler.
# (2026-07-25, extension of decision experiment #71.)
#
# PURPOSE — trace the frozen-backbone depth -> fidelity curve and, crucially,
# add the j=12 FROZEN point so that:
#   flagship j12 (+distilled LoRA)  vs  j12 FROZEN (no LoRA)
#   = SAME split depth, ONLY var = LoRA  -> isolates distilled LoRA's PURE
#     contribution at the flagship's own depth (decouples it from the 9->12
#     depth change that the #65 j9-frozen comparison conflates).
# Existing frozen no-LoRA points already measured: j0 (=#71) and j9 (=#65).
# Together {j0, j6, j9, j12} give a clean monotone depth-sweep (supports the
# bottleneck-layer-sweep-monotone finding on LoCoMo judge + BABILong).
#
# FIXED config per j (single variable vs flagship = resume_j + NO lora):
#   NO --lora_adapter | selector iter_bm25 (hop topk 4) | topk 12 | sink bos
#   | chunk_size 512 | NO --use_chat_template (chat=False)
#   model models/Qwen3-8b-local
#
# Coverage per j: LoCoMo (full, 8 shards) + BABILong {qa1,qa2,qa5} x
#   {0k,1k,2k,4k,8k,16k,32k} x 4 shards.  Scheduler = flock task-pool
#   (per-job .done -> resumable), identical to #71.
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
JVALUES="${JVALUES:-6 12}"

SELECTOR=iter_bm25
TOPK=12
ITER_HOP_TOPK=4
SINK=bos
CHUNK=512

LOGDIR="logs/qcmem_adapterfree_jsweep_chatFALSE"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

# ---------------------------------------------------------------------------
# Build the job pool. Line format (delimiter '|'):
#   BENCH | J | ATASK | ALEN | SHARD | NSHARD | OUTNAME
# ---------------------------------------------------------------------------
: > "$QUEUE"
for J in $JVALUES; do
  # (1) LoCoMo (full doc pack, heaviest -> emit first)
  for si in 0 1 2 3 4 5 6 7; do
    echo "locomo|$J|-|-|$si|8|-" >> "$QUEUE"
  done
  # (2) BABILong (short, <=32k)
  for L in 0k 1k 2k 4k 8k 16k 32k; do
    for si in 0 1 2 3; do
      echo "babilong|$J|qa1 qa2 qa5|$L|$si|4|qcmem_8b_zeroshot_j${J}_frozen_iterbm25_chatFALSE_$L" >> "$QUEUE"
    done
  done
done
NTASKS=$(wc -l < "$QUEUE")
echo "[$(date)] adapter-free j-sweep task-pool built: $NTASKS jobs | model=$MODEL | j in {$JVALUES} selector=$SELECTOR topk=$TOPK hop=$ITER_HOP_TOPK sink=$SINK chunk=$CHUNK | NO lora, NO chat_template | 8 GPU workers"

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
  local g="$1" bench="$2" j="$3" atask="$4" alen="$5" shard="$6" nshard="$7" outname="$8"
  local jobid; jobid="$(echo "${bench}_j${j}_${atask}_${alen}_s${shard}of${nshard}_${outname}" | tr ' /' '__')"
  local marker="$DONEDIR/$jobid.done"
  local log="$LOGDIR/$jobid.log"
  if [ -f "$marker" ]; then echo "[skip] GPU$g $jobid (done)"; return 0; fi
  local common="--resume_j $j --selector $SELECTOR --topk $TOPK --iter_hop_topk $ITER_HOP_TOPK --sink_tokens $SINK --chunk_size $CHUNK"
  echo "[$(date)] GPU$g START $jobid"
  local rc=0
  case "$bench" in
    locomo)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_locomo.py \
        --model_path "$MODEL" $common \
        --locomo_data locomo/data/locomo10.json --max_samples -1 \
        --output_dir "locomo_results/qcmem_8b_zeroshot_j${j}_frozen_iterbm25_chatFALSE" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    babilong)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_babilong.py \
        --model_path "$MODEL" $common \
        --tasks $atask --lengths $alen --limit 100 \
        --results_folder "babilong_results/qcmem_8b_zeroshot_j${j}_frozen_iterbm25_chatFALSE" --output_name "$outname" \
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
    IFS='|' read -r BENCH J ATASK ALEN SHARD NSHARD OUTNAME <<< "$line"
    run_job "$g" "$BENCH" "$J" "$ATASK" "$ALEN" "$SHARD" "$NSHARD" "$OUTNAME"
  done
  echo "[$(date)] GPU$g drained — queue empty"
}

PIDS=()
for g in "${GPUS[@]}"; do gpu_worker "$g" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done

# ---------------------------------------------------------------------------
# Scoring (official scorers only). LoCoMo GPT-4o judge run separately (CPU+API).
# ---------------------------------------------------------------------------
SUMMARY="$LOGDIR/SUMMARY.txt"
{
  echo "======= CoMem adapter-free j-sweep (frozen, iter_bm25, chat=False) SUMMARY ======="
  echo "date: $(date)"
  echo "config: NO-lora selector=$SELECTOR topk=$TOPK hop=$ITER_HOP_TOPK sink=$SINK chunk=$CHUNK chat=False | j in {$JVALUES}"
  for J in $JVALUES; do
    echo
    echo "########## j=$J  BABILong (compare_answers) ##########"
    $PYBIN scripts/score_nested_babilong.py "babilong_results/qcmem_8b_zeroshot_j${J}_frozen_iterbm25_chatFALSE" --expect -1 2>&1
    echo "########## j=$J  LoCoMo (F1; GPT-4o judge run separately) ##########"
    $PYBIN scripts/eval_qcmem_locomo.py --score_only --output_dir "locomo_results/qcmem_8b_zeroshot_j${J}_frozen_iterbm25_chatFALSE" 2>&1
  done
  echo
  echo "======= END SUMMARY $(date) ======="
} | tee "$SUMMARY"

touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] SCHED_DONE — summary at $SUMMARY"
