#!/usr/bin/env bash
# ============================================================================
# CoMem ADAPTER-FREE (training-free) — chat=False campaign, 8-GPU task-pool
# dynamic scheduler (2026-07-24, task #65).
#
# The adapter-free deployment point of CoMem: a FROZEN Qwen3-8B backbone with
# NO LoRA, resumed at the readout-safe split resume_j=9 (vs the distilled
# flagship's j=12). ONE fixed config across ALL benchmarks — j is NOT tuned
# per benchmark. This produces the "CoMem (adapter-free)" row that sits beside
# "CoMem (+ distilled LoRA)" in every main table, cell-for-cell comparable to
# the baseline chat=False rows (InfLLM #63, etc.).
#
# FIXED config (all 5 benchmarks):
#   resume_j 9 | NO --lora_adapter | selector iter_bm25 (hop topk 4) | topk 12
#   | sink bos | chunk_size 512 | NO --use_chat_template (no-think implied)
#   model models/Qwen3-8b-local
#
# adapter-free == simply OMIT --lora_adapter (there is NO --zero_training_no_adapter
# flag in any eval script — verified 2026-07-24). resume_j 9 gives the shallow
# readout-safe split; no weights are touched.
#
# Coverage (mirrors InfLLM #63 chat=False so the rows are cell-comparable, + LoCoMo):
#   RULER main : {niah_single(->niah_single_2), niah_multi(->niah_multikey_1),
#                 vt(->variable_tracking)} x {8k,16k,32k,64k,128k}  limit=100  8 shards
#   RULER tb   : {niah_single_1, niah_single_3, niah_multivalue, niah_multiquery}
#                 x {16k,64k,128k}  limit=100  8 shards
#   LongEval   : {8k,16k,32k,64k,128k}  num_samples=50  max_new_tokens=48  8 shards
#   LongBench  : {narrativeqa,qasper,hotpotqa,2wikimqa,multifieldqa_en,musique}
#                 all samples  8 shards
#   BABILong   : {qa1,qa2,qa5} x {0k,1k,2k,4k,8k,16k,32k}  limit=100  4 shards
#   LoCoMo     : full (max_samples=-1)  8 shards  (F1 now; GPT-4o judge later, same API key)
#
# Scheduler: 8 independent GPU workers each loop popping a job from a shared
# flock-protected queue -> single-GPU jobs, dynamic load balance, long 128k jobs
# never stall other GPUs. Queue emitted LPT-style (128k-heavy jobs first). A
# per-job .done marker makes reruns resumable.
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

# ---- FIXED adapter-free CoMem config (identical across all benchmarks) ----
RESUME_J=9
SELECTOR=iter_bm25
TOPK=12
ITER_HOP_TOPK=4
SINK=bos
CHUNK=512
COMMON="--resume_j $RESUME_J --selector $SELECTOR --topk $TOPK --iter_hop_topk $ITER_HOP_TOPK --sink_tokens $SINK --chunk_size $CHUNK"
# NOTE: no --lora_adapter (=> adapter-free) and no --use_chat_template (=> chat=False)

LOGDIR="logs/qcmem_adapterfree_j9_chatFALSE"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

RULER_MAIN_DIR="ruler_results/qcmem_8b_zeroshot_j9_chatFALSE"
RULER_TB_DIR="ruler_results/qcmem_8b_zeroshot_j9_taskbreadth_chatFALSE"
LE_DIR="longeval_results/qcmem_8b_zeroshot_j9_chatFALSE"          # results_folder/output_name split below
LB_DIR="longbench_results/qcmem_8b_zeroshot_j9_chatFALSE"
BL_DIR="babilong_results/qcmem_8b_zeroshot_j9_chatFALSE"
LC_DIR="locomo_results/qcmem_8b_zeroshot_j9_chatFALSE"
LE_NAME="qcmem_8b_zeroshot_j9_chatFALSE"

# ---------------------------------------------------------------------------
# Build the job pool. Line format (delimiter '|'):
#   BENCH | ATASK | ALEN | SHARD | NSHARD | OUTNAME
# ATASK/ALEN/OUTNAME may hold spaces — '|' is the ONLY field separator.
# ---------------------------------------------------------------------------
: > "$QUEUE"
# (1) RULER main (128k -> longest, emit first)
for t in niah_single niah_multi vt; do
  for si in 0 1 2 3 4 5 6 7; do
    echo "ruler|$t|8k 16k 32k 64k 128k|$si|8|qcmem_8b_zeroshot_j9_chatFALSE" >> "$QUEUE"
  done
done
# (2) RULER task-breadth (128k)
for t in niah_single_1 niah_single_3 niah_multivalue niah_multiquery; do
  for si in 0 1 2 3 4 5 6 7; do
    echo "ruler|$t|16k 64k 128k|$si|8|qcmem_8b_zeroshot_j9_taskbreadth_chatFALSE" >> "$QUEUE"
  done
done
# (3) LongEval (128k)
for si in 0 1 2 3 4 5 6 7; do
  echo "longeval|-|8k 16k 32k 64k 128k|$si|8|-" >> "$QUEUE"
done
# (4) LoCoMo (full doc pack, medium-heavy)
for si in 0 1 2 3 4 5 6 7; do
  echo "locomo|-|-|$si|8|-" >> "$QUEUE"
done
# (5) LongBench (medium)
for ds in narrativeqa qasper hotpotqa 2wikimqa multifieldqa_en musique; do
  for si in 0 1 2 3 4 5 6 7; do
    echo "longbench|$ds|-|$si|8|-" >> "$QUEUE"
  done
done
# (6) BABILong (short, emit last)
for L in 0k 1k 2k 4k 8k 16k 32k; do
  for si in 0 1 2 3; do
    echo "babilong|qa1 qa2 qa5|$L|$si|4|qcmem_8b_zeroshot_j9_chatFALSE_$L" >> "$QUEUE"
  done
done
NTASKS=$(wc -l < "$QUEUE")
echo "[$(date)] adapter-free task-pool built: $NTASKS jobs | model=$MODEL | resume_j=$RESUME_J selector=$SELECTOR topk=$TOPK hop=$ITER_HOP_TOPK sink=$SINK chunk=$CHUNK | NO lora, NO chat_template | 8 GPU workers"

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
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_ruler_qcmem.py \
        --model_path "$MODEL" $COMMON \
        --ruler_tasks $atask --lengths $alen --limit 100 \
        --results_folder ruler_results --output_name "$outname" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 --max_new_tokens 48 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    longeval)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_longeval.py \
        --model_path "$MODEL" $COMMON \
        --lengths $alen --num_samples 50 --max_new_tokens 48 \
        --results_folder longeval_results --output_name "$LE_NAME" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    locomo)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_locomo.py \
        --model_path "$MODEL" $COMMON \
        --locomo_data locomo/data/locomo10.json --max_samples -1 \
        --output_dir "$LC_DIR" \
        --num_shards "$nshard" --shard_index "$shard" --device cuda:0 \
        </dev/null >"$log" 2>&1; rc=$? ;;
    longbench)
      CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_qcmem_longbench.py \
        --model_path "$MODEL" $COMMON \
        --tasks $atask --max_samples -1 --hf_dataset data/longbench_raw/data \
        --output_dir "$LB_DIR" \
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
# Scoring (official scorers only — no re.search).
# ---------------------------------------------------------------------------
SUMMARY="$LOGDIR/SUMMARY.txt"
{
  echo "======= CoMem adapter-free (j9, chat=False) campaign SUMMARY ======="
  echo "date: $(date)"
  echo "config: resume_j=$RESUME_J NO-lora selector=$SELECTOR topk=$TOPK hop=$ITER_HOP_TOPK sink=$SINK chunk=$CHUNK chat=False"
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
  echo "########## LongEval (line-retrieval acc) ##########"
  $PYBIN scripts/eval_qcmem_longeval.py --score_only \
    --results_folder longeval_results --output_name "$LE_NAME" \
    --lengths 8k 16k 32k 64k 128k 2>&1
  echo
  echo "########## LongBench (qa_f1) ##########"
  $PYBIN scripts/eval_qcmem_longbench.py --score_only --output_dir "$LB_DIR" \
    --hf_dataset data/longbench_raw/data \
    --tasks narrativeqa qasper hotpotqa 2wikimqa multifieldqa_en musique 2>&1
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
