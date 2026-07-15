#!/usr/bin/env bash
# Fixed-protocol Qwen3-32B QCMem formal-eval pool. One model process per GPU.
# Usage: BENCHMARK=ruler|babilong MODEL_PATH=... [OUT_ROOT=...] bash $0
set -uo pipefail
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT"
BENCHMARK="${BENCHMARK:?ruler|babilong}"
MODEL_PATH="${MODEL_PATH:?stock Qwen3-32B path required}"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
BABILONG_DATASET_NAME="${BABILONG_DATASET_NAME:-$ROOT/data/babilong-hf}"
LIMIT=500; RESUME_J=16; CHUNK_SIZE=512; DTYPE=bfloat16; ATTN_IMPL=sdpa
if [ "$BENCHMARK" = ruler ]; then
  OUT_ROOT="${OUT_ROOT:-ruler_results/qwen32_zerotrain_n500_j16_chunk512}"
elif [ "$BENCHMARK" = babilong ]; then
  OUT_ROOT="${OUT_ROOT:-babilong_results/qwen32_zerotrain_n500_j16_chunk512}"
else
  echo "BENCHMARK must be ruler or babilong" >&2; exit 2
fi
LOG_ROOT="${LOG_ROOT:-logs/qwen32_zerotrain_n500_j16_chunk512/$BENCHMARK}"
POOL="$OUT_ROOT/.pool"; QUEUE="$POOL/queue.txt"; LOCK="$POOL/queue.lock"
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$POOL"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT:$ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.hf_cache/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Deliberately interleave long and short cells, then interleave shard indices so
# the initial eight jobs cover eight different cells. Queue unit=(cell, shard):
# 13*4=52 RULER jobs or 21*4=84 BABILong jobs.
CELLS="$POOL/cells.txt"; : >"$CELLS"; : >"$QUEUE"
if [ "$BENCHMARK" = ruler ]; then
  cat >>"$CELLS" <<'EOF'
niah_single 128k bm25 12 0 2
niah_multikey 8k bm25 12 0 2
vt 32k iter_bm25 16 4 4
niah_single 16k bm25 12 0 2
niah_multikey 64k bm25 12 0 2
vt 8k iter_bm25 16 4 4
niah_single 32k bm25 12 0 2
niah_multikey 128k bm25 12 0 2
vt 16k iter_bm25 16 4 4
niah_single 8k bm25 12 0 2
niah_multikey 32k bm25 12 0 2
niah_single 64k bm25 12 0 2
niah_multikey 16k bm25 12 0 2
EOF
else
  cat >>"$CELLS" <<'EOF'
qa1 32k bm25 12 0 2
qa5 0k bm25 12 0 2
qa2 16k bm25 12 0 2
qa1 1k bm25 12 0 2
qa5 8k bm25 12 0 2
qa2 2k bm25 12 0 2
qa1 4k bm25 12 0 2
qa5 32k bm25 12 0 2
qa2 0k bm25 12 0 2
qa1 16k bm25 12 0 2
qa5 1k bm25 12 0 2
qa2 8k bm25 12 0 2
qa1 2k bm25 12 0 2
qa5 4k bm25 12 0 2
qa2 32k bm25 12 0 2
qa1 0k bm25 12 0 2
qa5 16k bm25 12 0 2
qa2 1k bm25 12 0 2
qa1 8k bm25 12 0 2
qa5 2k bm25 12 0 2
qa2 4k bm25 12 0 2
EOF
fi
for si in 0 1 2 3; do while read -r cell; do echo "$cell $si" >>"$QUEUE"; done <"$CELLS"; done
echo "[$(date -Is)] $BENCHMARK pool queued $(wc -l <"$QUEUE") (cell,shard) jobs" | tee -a "$LOG_ROOT/pool.log"

pop_job() {
  local line
  exec 9>"$LOCK"; flock 9
  line="$(head -n1 "$QUEUE")"
  if [ -n "$line" ]; then tail -n+2 "$QUEUE" >"$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"; fi
  flock -u 9; exec 9>&-
  echo "$line"
}

worker() {
  local gpu="$1" line task len sel topk rounds hop si canonical name log rc failures=0
  while line="$(pop_job)"; [ -n "$line" ]; do
    read -r task len sel topk rounds hop si <<<"$line"
    canonical="$task"
    [ "$task" = niah_single ] && canonical=niah_single_2
    [ "$task" = niah_multikey ] && canonical=niah_multikey_1
    [ "$task" = vt ] && canonical=variable_tracking
    name="qwen32_zerotrain_${task}_${len}"
    log="$LOG_ROOT/${name}_shard${si}of4.log"
    if "$PY" scripts/qwen32_zerotrain_results.py --is-complete \
         "$BENCHMARK" "$OUT_ROOT" "$canonical" "$len" "$si" >/dev/null 2>&1; then
      echo "[$(date -Is)] gpu$gpu skip complete $task/$len shard$si"; continue
    fi
    echo "[$(date -Is)] gpu$gpu start $task/$len shard$si selector=$sel topk=$topk" | tee -a "$LOG_ROOT/pool.log"
    if [ "$BENCHMARK" = ruler ]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_ruler_qcmem.py \
          --model_path "$MODEL_PATH" --resume_j "$RESUME_J" --selector "$sel" --topk "$topk" \
          --iter_rounds "$rounds" --iter_hop_topk "$hop" --sink_tokens bos \
          --ruler_tasks "$task" --lengths "$len" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
          --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" --device cuda:0 \
          --num_shards 4 --shard_index "$si" \
          --results_folder "$OUT_ROOT" --output_name "$name" >"$log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_qcmem_babilong.py \
          --model_path "$MODEL_PATH" --resume_j "$RESUME_J" --selector bm25 --topk 12 \
          --sink_tokens bos --tasks "$task" --lengths "$len" --limit "$LIMIT" \
          --chunk_size "$CHUNK_SIZE" --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" --device cuda:0 \
          --dataset_name "$BABILONG_DATASET_NAME" \
          --num_shards 4 --shard_index "$si" \
          --results_folder "$OUT_ROOT" --output_name "$name" >"$log" 2>&1
    fi
    rc=$?; [ "$rc" -ne 0 ] && failures=$((failures+1))
    echo "[$(date -Is)] gpu$gpu end rc=$rc $task/$len shard$si" | tee -a "$LOG_ROOT/pool.log"
  done
  [ "$failures" -eq 0 ]
}

pids=()
for gpu in ${GPUS:-0 1 2 3 4 5 6 7}; do worker "$gpu" & pids+=("$!"); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
touch "$POOL/SCHED_DONE"
exit "$rc"
