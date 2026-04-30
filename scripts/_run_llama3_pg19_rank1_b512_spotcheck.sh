#!/usr/bin/env bash
# Llama-3-8B Q-Filters rank=1 pg19 spot-check at kv=512 / recent=64.
#
# Motivation: 2026-04-26 §11.4.3 WikiText rank sweep on Llama-3 found
# rank=1 strictly dominant (PPL 8.57 vs rank=2 21.75 vs rank=4 38.10 vs
# rank=8 89.38) at kv=512 / recent=64. On pg19 at the same op-point we
# only have rank=2 (PPL 28.28 from the original kv-curve). The §11.4.3
# cross-dataset synthesis is not load-bearing until we verify whether
# rank=1 also wins on pg19. This run supplies that single point.
#
# Hyperparameters match the rank=1 kv-extension sweep exactly (same
# calibration set, same recent window, same rope/bf16/sdpa). Reuses the
# cached rank=1 filters at kv=1024/rw=64 (filter cache is independent of
# kv_budget — it is a function of filter_rank + calibration_chunks only).
#
# Target: b200-1 (idle 8/8 per 15:10 audit). Est. ~3.5 min.
#
# CANONICAL WORKING DIRECTORY: /apdcephfs_wzc1/share_303098609/pighzliu_code
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
BASEFILTERS=outputs/rank1_kv_ext_llama3/qf_r1_b1024_rw64/filters.pt
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 1 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama3_pg19_rank1_b512_spot_${TS}.log
mkdir -p logs

echo "=== Llama-3 pg19 rank=1 kv=512/r64 spot-check starting $(date) ===" | tee "$ROOT_LOG"

OUT=outputs/pg19_llama3_rank1_spot/qf_r1_b512_rw64
LOG=logs/llama3_pg19_rank1_b512_eval_${TS}.log
mkdir -p "$OUT"

torchrun --nproc_per_node=8 --master_port=29551 scripts/eval_qfilters.py \
  --model "$MODEL" --data "$DATA" $COMMON \
  --kv_budget 512 --recent_window 64 --mode qfilters \
  --filters_cache "$BASEFILTERS" \
  --output_dir "$OUT" >> "$LOG" 2>&1

PPL=$(python -c "import json;print(json.load(open('${OUT}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
echo "[qf_r1_b512_rw64] rank=1 kv=512 recent=64 pg19 Llama-3 -> PPL=${PPL}" | tee -a "$ROOT_LOG"

echo "=== DONE $(date) ===" | tee -a "$ROOT_LOG"
