#!/usr/bin/env bash
# Paper A #62 — finish the Qwen3 model-family scale RULER table (chat=False).
# Remaining sizes after 4B/14B/32B: 0.6B (j=2, all cells), 1.7B (j=3, only the
# missing/truncated shards), 30B-A3B MoE (j=12, all cells).
#
# One shared 8-GPU worker pool over a combined (size, cell, shard) queue. Exact
# same fixed protocol as scripts/_qwen_scale_zerotrain_ruler_pool.sh:
#   niah_single/niah_multikey : selector=bm25 topk=12 rounds=0
#   vt (variable_tracking)    : selector=iter_bm25 topk=16 rounds=4 hop=4
#   chat=False, sink=bos, chunk=512, dtype=bf16, sdpa, LIMIT=500, num_shards=4.
# Only RESUME_J and the output prefix vary per size (per-size split depth j).
#
# Completeness gate is ROW-COUNT based (125 rows/shard for n=500, 4 shards) plus
# the per-shard _summary json — NOT the qwen32 gate, which hardcodes j==16 and
# 64 layers and would therefore never skip a scale cell (redoing finished work).
set -uo pipefail
ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$ROOT"
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LIMIT=500; CHUNK_SIZE=512; DTYPE=bfloat16; ATTN_IMPL=sdpa; NUM_SHARDS=4
MAX_RETRIES="${MAX_RETRIES:-2}"
EXPECT_ROWS=$((LIMIT / NUM_SHARDS))   # 125

export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.hf_cache/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

POOL_ROOT="${POOL_ROOT:-ruler_results/_p62_scale_pool}"
mkdir -p "$POOL_ROOT"
QUEUE="$POOL_ROOT/queue.txt"; LOCK="$POOL_ROOT/queue.lock"
REMAINING="$POOL_ROOT/remaining.txt"; FAILED="$POOL_ROOT/failed_jobs.txt"
POOLLOG="$POOL_ROOT/pool.log"

# size spec: label model resume_j out_root prefix
ALL_SIZES=(
  "0p6b    models/Qwen3-0.6B     2  ruler_results/qcmem_scale_0p6b_chatFALSE_ruler    qcmem_scale_0p6b"
  "1p7b    models/Qwen3-1.7B     3  ruler_results/qcmem_scale_1p7b_chatFALSE_ruler    qcmem_scale_1p7b"
  "30ba3b  models/Qwen3-30B-A3B  12 ruler_results/qcmem_scale_30ba3b_chatFALSE_ruler  qcmem_scale_30ba3b"
)
# SIZE_LABELS selects which sizes this pool runs (space-separated labels).
# Default = all. Used to split work across nodes (e.g. .104=30ba3b, .73=0p6b 1p7b).
SIZE_LABELS="${SIZE_LABELS:-0p6b 1p7b 30ba3b}"
SIZES=()
for spec in "${ALL_SIZES[@]}"; do
  read -r lbl _rest <<<"$spec"
  for want in $SIZE_LABELS; do
    [ "$lbl" = "$want" ] && SIZES+=("$spec")
  done
done

# 13 RULER cells: "task len sel topk rounds hop"
read -r -d '' CELLS <<'EOF'
niah_single   128k bm25      12 0 2
niah_multikey 8k   bm25      12 0 2
vt            32k  iter_bm25 16 4 4
niah_single   16k  bm25      12 0 2
niah_multikey 64k  bm25      12 0 2
vt            8k   iter_bm25 16 4 4
niah_single   32k  bm25      12 0 2
niah_multikey 128k bm25      12 0 2
vt            16k  iter_bm25 16 4 4
niah_single   8k   bm25      12 0 2
niah_multikey 32k  bm25      12 0 2
niah_single   64k  bm25      12 0 2
niah_multikey 16k  bm25      12 0 2
EOF

canon_of() {
  case "$1" in
    niah_single)   echo niah_single_2 ;;
    niah_multikey) echo niah_multikey_1 ;;
    vt)            echo variable_tracking ;;
    *)             echo "$1" ;;
  esac
}

# is_done <out_root> <prefix> <task> <len> <si>
is_done() {
  local out_root="$1" prefix="$2" task="$3" len="$4" si="$5"
  local canon; canon="$(canon_of "$task")"
  local celldir="$out_root/${prefix}_${task}_${len}"
  local csv="$celldir/${canon}_${len}_shard${si}of${NUM_SHARDS}.csv"
  local summ="$celldir/_summary_shard${si}of${NUM_SHARDS}.json"
  [ -f "$csv" ] || return 1
  [ -f "$summ" ] || return 1
  local rows; rows="$(tail -n +2 "$csv" 2>/dev/null | wc -l)"
  [ "$rows" = "$EXPECT_ROWS" ]
}

# --- build the combined queue (skip already-complete shards) ---
: >"$QUEUE"; : >"$FAILED"
built=0; skipped=0
# interleave shard index outer so the first 8 jobs hit 8 distinct cells
for si in 0 1 2 3; do
  for spec in "${SIZES[@]}"; do
    read -r label model rj out_root prefix <<<"$spec"
    while read -r task len sel topk rounds hop; do
      [ -z "$task" ] && continue
      if is_done "$out_root" "$prefix" "$task" "$len" "$si"; then
        skipped=$((skipped+1)); continue
      fi
      echo "$label $model $rj $out_root $prefix $task $len $sel $topk $rounds $hop $si 0" >>"$QUEUE"
      built=$((built+1))
    done <<<"$CELLS"
  done
done
echo "$built" >"$REMAINING"
echo "[$(date -Is)] p62 scale pool: queued $built jobs (skipped $skipped complete)" | tee -a "$POOLLOG"
if [ "${POOL_DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN jobs=$built skipped=$skipped"; cat "$QUEUE"; exit 0
fi
[ "$built" -eq 0 ] && { echo "[$(date -Is)] nothing to do" | tee -a "$POOLLOG"; touch "$POOL_ROOT/SCHED_DONE"; exit 0; }

pop_job() {
  local line
  exec 9>"$LOCK"; flock 9
  line="$(head -n1 "$QUEUE")"
  if [ -n "$line" ]; then tail -n+2 "$QUEUE" >"$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"; fi
  flock -u 9; exec 9>&-
  echo "$line"
}
remaining_jobs() { local n; exec 9>"$LOCK"; flock 9; n="$(cat "$REMAINING")"; flock -u 9; exec 9>&-; echo "$n"; }
mark_terminal() { local n; exec 9>"$LOCK"; flock 9; n="$(cat "$REMAINING")"; echo $((n-1)) >"$REMAINING"; flock -u 9; exec 9>&-; }
requeue_job() { exec 9>"$LOCK"; flock 9; echo "$1" >>"$QUEUE"; flock -u 9; exec 9>&-; }

worker() {
  local gpu="$1" line label model rj out_root prefix task len sel topk rounds hop si attempt
  local canon name log rc failures=0
  while true; do
    line="$(pop_job)"
    if [ -z "$line" ]; then
      [ "$(remaining_jobs)" -eq 0 ] && break
      sleep 2; continue
    fi
    read -r label model rj out_root prefix task len sel topk rounds hop si attempt <<<"$line"
    canon="$(canon_of "$task")"
    name="${prefix}_${task}_${len}"
    local logdir="logs/${prefix}_chatFALSE"; mkdir -p "$logdir"
    log="$logdir/${name}_shard${si}of${NUM_SHARDS}.log"
    if is_done "$out_root" "$prefix" "$task" "$len" "$si"; then
      echo "[$(date -Is)] gpu$gpu skip complete $label $task/$len shard$si" | tee -a "$POOLLOG"
      mark_terminal; continue
    fi
    echo "[$(date -Is)] gpu$gpu START $label $task/$len shard$si j=$rj sel=$sel topk=$topk" | tee -a "$POOLLOG"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_ruler_qcmem.py \
        --model_path "$model" --resume_j "$rj" --selector "$sel" --topk "$topk" \
        --iter_rounds "$rounds" --iter_hop_topk "$hop" --sink_tokens bos \
        --ruler_tasks "$task" --lengths "$len" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
        --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" --device cuda:0 \
        --num_shards "$NUM_SHARDS" --shard_index "$si" \
        --results_folder "$out_root" --output_name "$name" >"$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && is_done "$out_root" "$prefix" "$task" "$len" "$si"; then
      mark_terminal
      echo "[$(date -Is)] gpu$gpu END rc=0 $label $task/$len shard$si" | tee -a "$POOLLOG"
    elif [ "$attempt" -lt "$MAX_RETRIES" ]; then
      echo "[$(date -Is)] gpu$gpu RETRY rc=$rc attempt=$attempt $label $task/$len shard$si" | tee -a "$POOLLOG"
      requeue_job "$label $model $rj $out_root $prefix $task $len $sel $topk $rounds $hop $si $((attempt+1))"
    else
      echo "$label $task $len shard$si rc=$rc attempts=$((attempt+1))" >>"$FAILED"
      mark_terminal; failures=$((failures+1))
      echo "[$(date -Is)] gpu$gpu FAILED rc=$rc $label $task/$len shard$si" | tee -a "$POOLLOG"
    fi
  done
  [ "$failures" -eq 0 ]
}

pids=()
for gpu in $GPUS; do worker "$gpu" & pids+=("$!"); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
touch "$POOL_ROOT/SCHED_DONE"
echo "[$(date -Is)] p62 scale pool ALL_DONE rc=$rc (failed: $(wc -l <"$FAILED"))" | tee -a "$POOLLOG"
exit "$rc"
