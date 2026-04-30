#!/usr/bin/env bash
# Llama-3.0-8B cross-family Q-Filters eval on POST-FIX baseline.
# Re-runs the 2026-04-25 16:47 Llama-3 chain after the double-label-shift bug
# fix. Same 2-step protocol: dense baseline + qfilters at 256/64 headline.
# Reuses the existing Llama-3 calibration: outputs/qfilters_llama3_full_bestpoint/filters.pt
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
BASEFILTERS=outputs/qfilters_llama3_full_bestpoint/filters.pt
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama3_postfix_${TS}.log
echo "=== Llama-3 POST-FIX two-step starting $(date) ===" | tee "$ROOT_LOG"

# (1) Dense through harness
OUT1=outputs/postfix_llama3/dense_4096
mkdir -p "$OUT1"
LOG1=logs/llama3_postfix_dense_${TS}.log
echo "=== [1/2] DENSE at $(date) ===" | tee -a "$ROOT_LOG"
torchrun --nproc_per_node=8 --master_port=29533 scripts/eval_qfilters.py \
  --model "$MODEL" --data "$DATA" $COMMON \
  --kv_budget 4096 --recent_window 64 --mode sliding_window \
  --filters_cache "$BASEFILTERS" \
  --output_dir "$OUT1" >> "$LOG1" 2>&1
PPL1=$(python -c "import json;print(json.load(open('$OUT1/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
echo "[dense_4096] PPL=$PPL1" | tee -a "$ROOT_LOG"

# (2) Q-Filters 256/64 headline
OUT2=outputs/postfix_llama3/qf_b256_r64
mkdir -p "$OUT2"
LOG2=logs/llama3_postfix_qf256_${TS}.log
echo "=== [2/2] Q-FILTERS 256/64 at $(date) ===" | tee -a "$ROOT_LOG"
torchrun --nproc_per_node=8 --master_port=29533 scripts/eval_qfilters.py \
  --model "$MODEL" --data "$DATA" $COMMON \
  --kv_budget 256 --recent_window 64 --mode qfilters \
  --filters_cache "$BASEFILTERS" \
  --output_dir "$OUT2" >> "$LOG2" 2>&1
PPL2=$(python -c "import json;print(json.load(open('$OUT2/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
echo "[qf_b256_r64] PPL=$PPL2" | tee -a "$ROOT_LOG"

echo "=== Llama-3 POST-FIX DONE $(date) ===" | tee -a "$ROOT_LOG"
