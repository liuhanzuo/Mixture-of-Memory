#!/usr/bin/env bash
# Sample-sharded BABILong eval — 8 GPUs run the SAME (task,length) cell in parallel,
# each GPU takes samples [shard::8], so all 8 finish together (no straggler GPU).
#
# WHY (vs eval_mem_space_babilong_fast.sh): the "fast" launcher shards by
# (task,length) CELL across GPUs — so the GPU assigned qa1_32k runs all 100 hard
# samples while the GPU assigned qa2_2k finishes in minutes and then SITS IDLE.
# End times differ by hours = wasted GPU. This launcher instead processes cells
# SEQUENTIALLY, and for each cell fans the 100 samples across all 8 GPUs
# (--num_shards 8 --shard_index g, stride slice [g::8]). Every GPU does ~13
# samples of the SAME difficulty → all finish ~together → next cell → no idle GPU.
#
# run_babilong_mem_space.py already supports this (--num_shards/--shard_index,
# per-shard CSV; score_nested_babilong.py globs {task}_{length}_*.csv and merges).
#
# Usage:
#   CKPT=... ADAPTER_CONFIG=... RESULTS=... OUTPREFIX=... bash scripts/eval_babilong_sample_sharded.sh
# Env (all have defaults):
#   CKPT, ADAPTER_CONFIG, RESULTS, OUTPREFIX, MODEL, TASKS, LENGTHS, LIMIT,
#   CHUNK_SIZE, DTYPE, ATTN_IMPL, MAX_NEW_TOKENS, GPUS, EXTRA_ARGS
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_cache/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # reduce KV-cache fragmentation OOM

PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL="${MODEL:-models/Meta-Llama-3-8B}"
CKPT="${CKPT:?set CKPT}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:?set ADAPTER_CONFIG}"
RESULTS="${RESULTS:?set RESULTS}"
OUTPREFIX="${OUTPREFIX:?set OUTPREFIX}"
TASKS="${TASKS:-qa1 qa2 qa5}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
LIMIT="${LIMIT:-100}"
BATCH_SIZE="${BATCH_SIZE:-auto}"   # "auto" = per-length bs (bs_for_len); or a fixed int.
                                   # L20A 183GB: short lengths bs8, 16k bs2, 32k bs1
                                   # (bs4@32k OOMs — 62 chunks/sample + KV fragmentation).
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOGDIR="${LOGDIR:-logs/eval_$(basename "$RESULTS")}"
mkdir -p "$RESULTS" "$LOGDIR"
read -r -a LENGTHS <<< "${LENGTHS:-0k 1k 2k 4k 8k 16k 32k}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NG=${#GPUS[@]}

echo "[$(date)] sample-sharded eval: ckpt=$CKPT NG=$NG shards/cell=$NG limit=$LIMIT"
echo "[$(date)] cells run SEQUENTIALLY; each fanned across $NG GPUs (samples [g::$NG])"

# Per-length batch size (auto). KEY FINDING 2026-07-04: bs=4 @ 32k OOM'd ONLY
# because expandable_segments was off (178GB, 90GB was fragmentation). WITH
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (exported above), bs=4 @ 32k
# uses just 57GB/183 — so long contexts have plenty of headroom too. These bs are
# tuned to fill ~half the 183GB with margin. Override any cell via BATCH_SIZE=int.
bs_for_len() {
  if [ "${BATCH_SIZE}" != "auto" ]; then echo "$BATCH_SIZE"; return; fi
  case "$1" in
    0k|1k|2k)   echo 16 ;;
    4k)         echo 12 ;;
    8k)         echo 8 ;;
    16k)        echo 6 ;;
    32k)        echo 4 ;;
    *)          echo 2 ;;
  esac
}

for task in $TASKS; do
  for L in "${LENGTHS[@]}"; do
    CELL_BS="$(bs_for_len "$L")"
    # Resume support: skip a cell whose expected shard CSVs are ALL present with
    # the right row counts. A shard i is "done" if its CSV has >=1 data row AND
    # the shard's expected sample count (ceil((limit-i)/NG)) matches. Cheap check:
    # if NG shard CSVs exist for this cell and total rows >= LIMIT, skip. Lets a
    # killed/restarted run avoid re-doing finished cells (SKIP_DONE=0 to force).
    if [ "${SKIP_DONE:-1}" = "1" ]; then
      celldir="$RESULTS/${OUTPREFIX}_${task}_${L}"
      ncsv=$(find "$celldir" -name "${task}_${L}_*.csv" 2>/dev/null | wc -l)
      rows=$(find "$celldir" -name "${task}_${L}_*.csv" 2>/dev/null -exec cat {} \; 2>/dev/null | grep -c '.' 2>/dev/null || echo 0)
      if [ "$ncsv" -ge "$NG" ] && [ "$rows" -ge "$LIMIT" ]; then
        echo "[$(date)] === CELL $task $L SKIP (already done: $ncsv shards, ~$rows rows) ==="
        continue
      fi
    fi
    echo "[$(date)] === CELL $task $L : launching $NG shards (one per GPU), bs=$CELL_BS ==="
    pids=()
    for ((g=0; g<NG; g++)); do
      GPU="${GPUS[$g]}"
      CUDA_VISIBLE_DEVICES="$GPU" $PYBIN scripts/run_babilong_mem_space.py \
        --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ADAPTER_CONFIG" \
        --results_folder "$RESULTS" --output_name "${OUTPREFIX}_${task}_${L}" \
        --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
        --batch_size "$CELL_BS" --max_new_tokens "$MAX_NEW_TOKENS" \
        --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" \
        --num_shards "$NG" --shard_index "$g" $EXTRA_ARGS \
        </dev/null >"$LOGDIR/${task}_${L}_shard${g}.log" 2>&1 &
      pids+=($!)
    done
    # Barrier: wait for ALL 8 shards of THIS cell before the next cell — so the
    # whole fleet is always working on one cell together, none idle.
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
    echo "[$(date)] === CELL $task $L done ($fail shard failures) ==="
  done
done
echo "[$(date)] ALL cells done -> $RESULTS  (score: $PYBIN scripts/score_nested_babilong.py $RESULTS)"
