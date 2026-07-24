#!/usr/bin/env bash
# ============================================================================
# InfLLM baseline — chat=False campaign, 8-GPU task-pool dynamic scheduler
# (2026-07-24). Runs the chat=False twin of the diskB `infllm_8b` chat=True
# results across 4 benchmarks {RULER, LongBench, LongEval, BABILong} with the
# SAME model / coverage / InfLLM config, the ONLY difference being the omission
# of --use_chat_template.
#
# Coverage (mirrors chat=True on diskB .73):
#   RULER main : {niah_single(->niah_single_2), niah_multi(->niah_multikey_1),
#                 vt(->variable_tracking)} x {8k,16k,32k,64k,128k}  limit=100  8 shards
#   RULER tb   : {niah_single_1, niah_single_3, niah_multivalue, niah_multiquery}
#                 x {16k,64k,128k}  limit=100  8 shards
#   LongBench  : {narrativeqa,qasper,hotpotqa,2wikimqa,multifieldqa_en,musique}
#                 all samples  8 shards
#   LongEval   : {8k,16k,32k,64k,128k}  limit=50  max_new_tokens=48  8 shards
#   BABILong   : {qa1,qa2,qa5} x {0k,1k,2k,4k,8k,16k,32k}  limit=100  4 shards
#
# Scheduler: 8 independent GPU workers each loop popping a job from a shared
# flock-protected queue -> single-GPU jobs, dynamic load balance, long 128k jobs
# never stall other GPUs. Queue is emitted LPT-style (128k-heavy jobs first) so
# the makespan tail is short jobs. A per-job .done marker makes reruns resumable.
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }

PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/external/InfLLM:$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"

MODEL="${MODEL:-models/Qwen3-8b-local}"
GPUS=(0 1 2 3 4 5 6 7)

LOGDIR="logs/infllm_chatFALSE_taskpool"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

RULER_MAIN_DIR="ruler_results/infllm_8b_chatFALSE"
RULER_TB_DIR="ruler_results/infllm_8b_taskbreadth_chatFALSE"
LB_DIR="longbench_results/infllm_8b_chatFALSE"
LE_DIR="longeval_results/infllm_8b_chatFALSE"
BL_DIR="babilong_results/infllm_8b_chatFALSE"

# ---------------------------------------------------------------------------
# Build the job pool. Line format (delimiter '|'):
#   BENCH | ATASK | ALEN | SHARD | NSHARD | OUTNAME
# ATASK / ALEN / OUTNAME may hold spaces (e.g. "8k 16k 32k") — '|' is the ONLY
# field separator, so internal spaces survive `IFS='|' read`.
# ---------------------------------------------------------------------------
: > "$QUEUE"
# (1) RULER main (has 128k -> longest, emit first)
for t in niah_single niah_multi vt; do
  for si in 0 1 2 3 4 5 6 7; do
    echo "ruler|$t|8k 16k 32k 64k 128k|$si|8|infllm_8b_chatFALSE" >> "$QUEUE"
  done
done
# (2) RULER task-breadth (has 128k)
for t in niah_single_1 niah_single_3 niah_multivalue niah_multiquery; do
  for si in 0 1 2 3 4 5 6 7; do
    echo "ruler|$t|16k 64k 128k|$si|8|infllm_8b_taskbreadth_chatFALSE" >> "$QUEUE"
  done
done
# (3) LongEval (has 128k)
for si in 0 1 2 3 4 5 6 7; do
  echo "longeval|-|8k 16k 32k 64k 128k|$si|8|-" >> "$QUEUE"
done
# (4) LongBench (medium)
for ds in narrativeqa qasper hotpotqa 2wikimqa multifieldqa_en musique; do
  for si in 0 1 2 3 4 5 6 7; do
    echo "longbench|$ds|-|$si|8|-" >> "$QUEUE"
  done
done
# (5) BABILong (short, emit last)
for L in 0k 1k 2k 4k 8k 16k 32k; do
  for si in 0 1 2 3; do
    echo "babilong|qa1 qa2 qa5|$L|$si|4|infllm_8b_chatFALSE_$L" >> "$QUEUE"
  done
done
NTASKS=$(wc -l < "$QUEUE")
echo "[$(date)] task-pool built: $NTASKS jobs | model=$MODEL | 8 GPU workers"

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
    ruler)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_infllm_ruler.py \
        --model_path "$MODEL" --ruler_tasks $atask --lengths $alen --limit 100 \
        --results_folder ruler_results --output_name "$outname" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    longbench)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_infllm_longbench.py \
        --model_path "$MODEL" --tasks $atask --hf_dataset data/longbench_raw/data \
        --output_dir "$LB_DIR" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    longeval)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_infllm_longeval.py \
        --model_path "$MODEL" --lengths $alen --limit 50 --max_new_tokens 48 \
        --output_dir "$LE_DIR" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    babilong)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_infllm_babilong.py \
        --model_path "$MODEL" --tasks $atask --lengths $alen --limit 100 \
        --results_folder "$BL_DIR" --output_name "$outname" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
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
# Scoring (official scorers only — no re.search).
# ---------------------------------------------------------------------------
SUMMARY="$LOGDIR/SUMMARY.txt"
{
  echo "================ InfLLM chat=False campaign SUMMARY ================"
  echo "date: $(date)"
  echo
  echo "########## RULER main (string_match) ##########"
  $PYBIN scripts/score_ruler_taskbreadth.py "$RULER_MAIN_DIR" \
    --tasks niah_single_2 niah_multikey_1 variable_tracking \
    --lengths 8k 16k 32k 64k 128k --num_shards 8 --limit 100 2>&1
  echo
  echo "########## RULER task-breadth (string_match) ##########"
  $PYBIN scripts/score_ruler_taskbreadth.py "$RULER_TB_DIR" \
    --tasks niah_single_1 niah_single_3 niah_multivalue niah_multiquery \
    --lengths 16k 64k 128k --num_shards 8 --limit 100 2>&1
  echo
  echo "########## LongBench (qa_f1) ##########"
  $PYBIN scripts/eval_infllm_longbench.py --score_only --output_dir "$LB_DIR" \
    --tasks narrativeqa qasper hotpotqa 2wikimqa multifieldqa_en musique 2>&1
  echo
  echo "########## LongEval (line-retrieval acc) ##########"
  $PYBIN scripts/eval_infllm_longeval.py --score_only --output_dir "$LE_DIR" \
    --lengths 8k 16k 32k 64k 128k 2>&1
  echo
  echo "########## BABILong (compare_answers) ##########"
  $PYBIN scripts/score_nested_babilong.py "$BL_DIR" --expect -1 2>&1
  echo
  echo "================ END SUMMARY $(date) ================"
} | tee "$SUMMARY"

touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] SCHED_DONE — summary at $SUMMARY"
