#!/usr/bin/env bash
# run.sh — turnkey driver: run all (method × task × length) baseline cells,
# sharded round-robin across the 8 GPUs of ONE node. Run this on EACH of your
# two nodes (they are independent; just run the same command on both — the
# dir-skip makes it safe to re-run / resume).
#
# Prereqs (once):
#   pip install "torch>=2.1" "transformers>=4.44" accelerate
#   # optional faster attn: pip install flash-attn --no-build-isolation
#   # download models once (needs internet / HF token for gated Llama):
#   #   huggingface-cli download Qwen/Qwen3-8B
#   #   huggingface-cli download meta-llama/Meta-Llama-3-8B
#
# Usage:
#   MODEL=Qwen/Qwen3-8B bash run.sh
#   MODEL=meta-llama/Meta-Llama-3-8B bash run.sh
set -u
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT="${OUT:-results/$(echo "$MODEL" | tr '/' '_')}"
NSAMP="${NSAMP:-100}"
WINDOW="${WINDOW:-4096}"          # StreamingLLM KV budget (match our QCMem read budget)
# 40GB A100: 'full' at 32k+ may OOM for 8B. Start with lengths that fit; add 64k/128k
# only if full-context doesn't OOM (streaming always fits since KV budget is fixed).
LENGTHS="${LENGTHS:-1k 2k 4k 8k 16k 32k}"
TASKS="${TASKS:-niah_single niah_multikey vt}"
METHODS="${METHODS:-full streaming}"
PY="${PY:-python}"
mkdir -p "$OUT"

# build the full cell list, then round-robin onto the 8 GPUs
cells=(); for m in $METHODS; do for t in $TASKS; do for l in $LENGTHS; do
  cells+=("$m|$t|$l"); done; done; done

echo "[run] model=$MODEL  ${#cells[@]} cells over 8 GPUs  out=$OUT"
gi=0
for c in "${cells[@]}"; do
  m="${c%%|*}"; rest="${c#*|}"; t="${rest%%|*}"; l="${rest##*|}"
  g=$((gi % 8))
  tag="${m}_${t}_${l}"
  if [ -f "$OUT/$tag.json" ]; then echo "  skip done $tag"; gi=$((gi+1)); continue; fi
  # serialize per-GPU by waiting when that GPU already has a job in this pass:
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" eval_ruler_baseline.py \
    --model_path "$MODEL" --method "$m" --task "$t" --length "$l" \
    --num_samples "$NSAMP" --window_tokens "$WINDOW" \
    --out_dir "$OUT" > "$OUT/log_${tag}.txt" 2>&1 &
  gi=$((gi+1))
  # after dispatching 8 (one per GPU), wait for them before the next batch
  if [ $((gi % 8)) -eq 0 ]; then wait; fi
done
wait
echo "[run] ALL DONE. Results (recall) summary:"
grep -h RESULT "$OUT"/log_*.txt 2>/dev/null | sort
echo "[run] send back the whole '$OUT' folder (the *.json summaries are enough for us)."
