#!/usr/bin/env bash
# ============================================================================
# Paper A P1.6 — standard SnapKV + PyramidKV KV-cache-compression baselines on
# Qwen3-8B at the CoMem read budget (max_capacity_prompt=6657), 8-GPU task-pool
# dynamic scheduler.
#
# DRY BY DEFAULT: with no env override this script only PRINTS the full command
# matrix (selftest gate + every RULER/LoCoMo job) and exits WITHOUT touching a
# GPU. Set DRY=0 to actually launch. (MAIN reviews the DRY output, then flips.)
#
#   Preview:   bash scripts/_run_p16_baselines.sh
#   Launch :   DRY=0 setsid nohup bash scripts/_run_p16_baselines.sh \
#                  >logs/p16_kvcompress/sched.out 2>&1 &
#
# Coverage (chat=False, enable_thinking=False, bf16, SDPA, greedy — Paper A):
#   selftest    : {snapkv,pyramidkv} faithfulness gate (no-op<budget + retained==budget)
#   RULER native: {snapkv,pyramidkv} x {niah_single_2,niah_multikey_1,variable_tracking}
#                   x {8k,16k,32k,64k,128k}  n=100  8 shards   (64k/128k left-trunc to 40960)
#   RULER yarn  : {snapkv,pyramidkv} x {3 tasks} x {64k,128k}  n=100  8 shards
#                   (full >40960 prompt via YaRN; reported as "<method>-yarn")
#   LoCoMo      : {snapkv,pyramidkv} full set  8 shards
#
# Why prefill-then-compress needs the instrumentation: these methods run the
# FULL exact prefill over the whole prompt THEN compress the STORED KV. The
# harness records per cell: quality, full-prefill latency, peak GPU mem,
# compressed retained-KV bytes + per-layer retained length, decode latency/tok,
# full_prompt_seen=True, OOM/fallback. (Contrast CoMem's persistent bounded read.)
#
# Scheduler: 8 GPU workers loop popping jobs from a shared flock-protected queue
# (LPT-ish: 128k-heavy jobs first). Per-job .done marker => resumable reruns.
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }

PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
DRY="${DRY:-1}"                      # 1 = print only (default), 0 = execute
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"

MODEL="${MODEL:-models/Qwen3-8b-local}"
BUDGET="${BUDGET:-6657}"             # CoMem read budget: BOS 1 + top-12*512 + query<=512
WINDOW="${WINDOW:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NSHARD="${NSHARD:-8}"
METHODS=(${METHODS:-snapkv pyramidkv})
RULER_TASKS=(${RULER_TASKS:-niah_single_2 niah_multikey_1 variable_tracking})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})

LOGDIR="logs/p16_kvcompress"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
QUEUE="$LOGDIR/task_queue.txt"
LOCK="$LOGDIR/queue.lock"

RULER_ROOT="ruler_results"
LOCOMO_ROOT="locomo_results"

echo "============================================================"
echo " P1.6 KV-compression baseline launcher"
echo "   PROJECT_ROOT = $PROJECT_ROOT"
echo "   PYBIN        = $PYBIN"
echo "   MODEL        = $MODEL"
echo "   BUDGET       = $BUDGET  (window=$WINDOW)"
echo "   methods      = ${METHODS[*]}"
echo "   RULER tasks  = ${RULER_TASKS[*]}"
echo "   n_samples    = $NUM_SAMPLES   shards=$NSHARD   GPUs=${GPUS[*]}"
echo "   DRY          = $DRY  (1=print only, 0=execute)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Build the job pool. Line format (delimiter '|'):
#   KIND | METHOD | TASKS | LENGTHS | LONGCTX | SHARD | NSHARD | OUTNAME
# TASKS / LENGTHS may hold spaces; '|' is the ONLY field separator.
# ---------------------------------------------------------------------------
: > "$QUEUE"
for m in "${METHODS[@]}"; do
  # (1) RULER native — all 5 lengths in one job (harness loops task x length).
  for t in "${RULER_TASKS[@]}"; do
    for si in $(seq 0 $((NSHARD-1))); do
      echo "ruler|$m|$t|8k 16k 32k 64k 128k|native|$si|$NSHARD|p16_${m}_native" >> "$QUEUE"
    done
  done
  # (2) RULER yarn — only the >native lengths, reported as <method>-yarn.
  for t in "${RULER_TASKS[@]}"; do
    for si in $(seq 0 $((NSHARD-1))); do
      echo "ruler|$m|$t|64k 128k|yarn|$si|$NSHARD|p16_${m}_yarn" >> "$QUEUE"
    done
  done
  # (3) LoCoMo full set.
  for si in $(seq 0 $((NSHARD-1))); do
    echo "locomo|$m|-|-|native|$si|$NSHARD|p16_${m}" >> "$QUEUE"
  done
done
NJOBS=$(wc -l < "$QUEUE")
echo "[$(date)] task-pool built: $NJOBS jobs"

# ---- render the exact command for a job line (used by both DRY and run) ----
render_cmd() {
  local g="$1" kind="$2" method="$3" tasks="$4" lengths="$5" longctx="$6" \
        shard="$7" nshard="$8" outname="$9"
  local common="--model_path $MODEL --method $method \
--max_capacity_prompt $BUDGET --window_size $WINDOW \
--num_shards $nshard --shard_index $shard --device cuda:0"
  case "$kind" in
    ruler)
      echo "CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_p16_kvcompress.py \
--mode ruler $common --tasks $tasks --lengths $lengths --long_ctx $longctx \
--num_samples $NUM_SAMPLES --results_folder $RULER_ROOT --output_name $outname" ;;
    locomo)
      echo "CUDA_VISIBLE_DEVICES=$g $PYBIN scripts/eval_p16_kvcompress.py \
--mode locomo $common --long_ctx $longctx --output_dir $LOCOMO_ROOT/$outname" ;;
  esac
}

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
  local g="$1" kind="$2" method="$3" tasks="$4" lengths="$5" longctx="$6" \
        shard="$7" nshard="$8" outname="$9"
  local jobid; jobid="$(echo "${kind}_${method}_${tasks}_${lengths}_${longctx}_s${shard}of${nshard}" | tr ' /' '__')"
  local marker="$DONEDIR/$jobid.done"
  local log="$LOGDIR/$jobid.log"
  if [ -f "$marker" ]; then echo "[skip] GPU$g $jobid (done)"; return 0; fi
  local cmd; cmd="$(render_cmd "$g" "$kind" "$method" "$tasks" "$lengths" "$longctx" "$shard" "$nshard" "$outname")"
  echo "[$(date)] GPU$g START $jobid"
  eval "$cmd </dev/null >\"$log\" 2>&1"; local rc=$?
  if [ "$rc" -eq 0 ]; then touch "$marker"; echo "[$(date)] GPU$g DONE  $jobid";
  else echo "[$(date)] GPU$g FAIL  $jobid rc=$rc (see $log)"; fi
}

# ---- a GPU worker: pop jobs until queue empty ----
gpu_worker() {
  local g="$1"; sleep $((g * 8))
  while true; do
    local line; line="$(pop_job)"
    [ -z "$line" ] && break
    IFS='|' read -r KIND METHOD TASKS LENGTHS LONGCTX SHARD NSH OUTNAME <<< "$line"
    run_job "$g" "$KIND" "$METHOD" "$TASKS" "$LENGTHS" "$LONGCTX" "$SHARD" "$NSH" "$OUTNAME"
  done
  echo "[$(date)] GPU$g drained — queue empty"
}

# ===========================================================================
# DRY: print the selftest gate + every job command, then exit.
# ===========================================================================
if [ "$DRY" != "0" ]; then
  echo
  echo "########## DRY RUN — commands only, no GPU touched ##########"
  echo
  echo "# --- faithfulness gate (run FIRST on 1 GPU before the campaign) ---"
  for m in "${METHODS[@]}"; do
    echo "CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYBIN scripts/eval_p16_kvcompress.py \
--mode selftest --method $m --model_path $MODEL \
--max_capacity_prompt $BUDGET --window_size $WINDOW \
--selftest_out $LOGDIR/selftest_${m}.json"
  done
  echo
  echo "# --- campaign jobs (dispatched round-robin over GPUs ${GPUS[*]}) ---"
  gi=0
  while IFS='|' read -r KIND METHOD TASKS LENGTHS LONGCTX SHARD NSH OUTNAME; do
    [ -z "$KIND" ] && continue
    g="${GPUS[$((gi % ${#GPUS[@]}))]}"; gi=$((gi+1))
    render_cmd "$g" "$KIND" "$METHOD" "$TASKS" "$LENGTHS" "$LONGCTX" "$SHARD" "$NSH" "$OUTNAME"
  done < "$QUEUE"
  echo
  echo "# --- post-hoc scoring (after all shards land) ---"
  for m in "${METHODS[@]}"; do
    echo "# RULER $m native:"
    echo "$PYBIN scripts/score_ruler_taskbreadth.py $RULER_ROOT/p16_${m}_native \
--tasks ${RULER_TASKS[*]} --lengths 8k 16k 32k 64k 128k --num_shards $NSHARD --limit $NUM_SAMPLES"
    echo "# RULER $m yarn (64k/128k):"
    echo "$PYBIN scripts/score_ruler_taskbreadth.py $RULER_ROOT/p16_${m}_yarn \
--tasks ${RULER_TASKS[*]} --lengths 64k 128k --num_shards $NSHARD --limit $NUM_SAMPLES"
    echo "# LoCoMo $m:"
    echo "$PYBIN scripts/eval_p16_kvcompress.py --mode aggregate --agg_kind locomo \
--output_dir $LOCOMO_ROOT/p16_${m}"
  done
  echo
  echo "[$(date)] DRY complete: $NJOBS campaign jobs + ${#METHODS[@]} selftests. Set DRY=0 to launch."
  exit 0
fi

# ===========================================================================
# EXECUTE: selftest gate first (fail-closed), then the 8-GPU pool, then score.
# ===========================================================================
echo "[$(date)] running faithfulness gate on GPU ${GPUS[0]} ..."
GATE_OK=1
for m in "${METHODS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYBIN scripts/eval_p16_kvcompress.py \
    --mode selftest --method "$m" --model_path "$MODEL" \
    --max_capacity_prompt "$BUDGET" --window_size "$WINDOW" \
    --selftest_out "$LOGDIR/selftest_${m}.json" \
    2>&1 | tee "$LOGDIR/selftest_${m}.log"
  if ! grep -q '"overall_pass": true' "$LOGDIR/selftest_${m}.json" 2>/dev/null; then
    echo "[$(date)] SELFTEST FAILED for $m — aborting campaign (see $LOGDIR/selftest_${m}.log)"
    GATE_OK=0
  fi
done
if [ "$GATE_OK" -ne 1 ]; then echo "[$(date)] gate failed, not launching pool"; exit 4; fi
echo "[$(date)] gate PASSED — launching $NJOBS jobs on ${#GPUS[@]} GPU workers"

PIDS=()
for g in "${GPUS[@]}"; do gpu_worker "$g" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
SUMMARY="$LOGDIR/SUMMARY.txt"
{
  echo "================ P1.6 KV-compression baseline SUMMARY ================"
  echo "date: $(date)  budget=$BUDGET window=$WINDOW  chat=False think=False"
  for m in "${METHODS[@]}"; do
    echo; echo "########## RULER $m native (string_match_all) ##########"
    $PYBIN scripts/score_ruler_taskbreadth.py "$RULER_ROOT/p16_${m}_native" \
      --tasks "${RULER_TASKS[@]}" --lengths 8k 16k 32k 64k 128k \
      --num_shards "$NSHARD" --limit "$NUM_SAMPLES" 2>&1
    echo; echo "########## RULER $m yarn 64k/128k ##########"
    $PYBIN scripts/score_ruler_taskbreadth.py "$RULER_ROOT/p16_${m}_yarn" \
      --tasks "${RULER_TASKS[@]}" --lengths 64k 128k \
      --num_shards "$NSHARD" --limit "$NUM_SAMPLES" 2>&1
    echo; echo "########## LoCoMo $m (F1/EM/acc) ##########"
    $PYBIN scripts/eval_p16_kvcompress.py --mode aggregate --agg_kind locomo \
      --output_dir "$LOCOMO_ROOT/p16_${m}" 2>&1
  done
  echo; echo "================ END SUMMARY $(date) ================"
} | tee "$SUMMARY"

touch "$LOGDIR/SCHED_DONE"
echo "[$(date)] SCHED_DONE — summary at $SUMMARY"
