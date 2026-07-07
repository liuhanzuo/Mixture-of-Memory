#!/usr/bin/env bash
# StreamingLLM (fixed KV budget) RULER baseline — 8-GPU sharded launcher.
#
# Runs scripts/eval_ruler_streamingllm.py (sink+window truncation approx of
# StreamingLLM) on Qwen3-8B, niah_single (_2, PG19-prose haystack), at the same
# lengths + n=50 sample set + string_match_all scoring as the QCMem RULER run,
# so the two are directly comparable. Budget sink=4 + window=6653 = 6657 ~=
# QCMem read length.
#
# Sharding: num_shards=8, each GPU builds the full 50-sample set (deterministic)
# but only generates for its shard's indices [i::8]. Merge with
# scripts/score_nested_babilong.py (sums per-shard CSVs into one cell).
#
# Node: diskB H20 (28.83.53.31:36000). Run detached via setsid.
set -uo pipefail
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

export HF_HUB_OFFLINE=1
export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_CACHE=$PWD/.hf_cache/datasets

PYBIN=.venv/bin/python
MODEL=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
RESULTS=ruler_results
OUT_NAME=streamingllm_qwen_niah
TASKS="niah_single"
LENGTHS="8k 16k 64k 128k"
LIMIT=50
SINK=4
WINDOW=6653
LOGDIR=logs/streamingllm_qwen
mkdir -p "$LOGDIR" "$RESULTS/$OUT_NAME"

echo "=== StreamingLLM RULER launch $(date) ==="
echo "model=$MODEL tasks=$TASKS lengths=$LENGTHS limit=$LIMIT sink=$SINK window=$WINDOW"
echo "outdir=$RESULTS/$OUT_NAME (per-shard CSVs: <task>_<len>_shard{g}of8.csv)"

for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g nohup $PYBIN scripts/eval_ruler_streamingllm.py \
    --model_path "$MODEL" \
    --sink_size $SINK --window_size $WINDOW \
    --ruler_tasks $TASKS --lengths $LENGTHS \
    --limit $LIMIT --num_shards 8 --shard_index $g \
    --max_new_tokens 48 --device cuda:0 --dtype bfloat16 \
    --output_name "$OUT_NAME" --results_folder "$RESULTS" \
    >"$LOGDIR/shard${g}.out" 2>&1 &
  echo "launched shard $g on GPU $g (pid $!)"
done

wait
echo "=== all shards done $(date) ==="
ls -la "$RESULTS/$OUT_NAME/" | head -60

# --- merge shards -> RULER string_match_all cell scores (mean of recall col) ---
echo "=== merged RULER string_match_all scores ==="
$PYBIN scripts/merge_streamingllm_ruler.py "$RESULTS/$OUT_NAME" \
  --lengths $LENGTHS --tasks niah_single_2 2>&1 | tee "$RESULTS/$OUT_NAME/_merged.txt"
echo "=== DONE ==="
