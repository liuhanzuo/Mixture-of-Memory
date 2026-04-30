#!/usr/bin/env bash
# Llama-3-8B Q-Filters STREAMING eval at seq_length=32k on pg19.
#
# Completes the last §11.4 checklist item in
# ops/research_notes/20260426_s11_retraction.md: "Streaming eval ≥ 32k — ⏳
# pending; no driver script yet".
#
# What this runs:
#   Two modes at the best Llama-3 op-point (§11.4.3 + pg19 rank=1 spot-check):
#     rank=1, kv_budget=512, recent_window=64, calibration_chunks=64,
#     sub_window_len=1024, bf16 + sdpa.
#   Streams are derived from data/pg19_chunks_llama3_noeos.npy by concatenating
#   8 consecutive 4096-token chunks → 32768 tokens per stream, 16 streams.
#   Reuses the already-cached rank=1 filters
#     outputs/rank1_kv_ext_llama3/qf_r1_b1024_rw64/filters.pt
#   (the filter file is a function of filter_rank + calibration_chunks only,
#   independent of kv_budget).
#
#   Comparison:
#     qfilters        — full Q-Filters compression.
#     sliding_window  — SWA baseline at the same kv_budget.
#   Both modes are "streaming": one continuous 32k-token document per stream,
#   cache carries across all 32 sub-windows, compression hook keeps it bounded
#   to kv_budget=512. RoPE positions via Patch A (re-rotation) stay ≤
#   kv_budget+sub_window_len, so no naive extrapolation in either mode.
#
#   We do NOT run a dense streaming baseline at 32k because Llama-3-8B trained
#   at 8192 — a dense 32k run would naively extrapolate RoPE and is not a
#   faithful "dense" point. The dense reference at 4096 in §11.2 is the right
#   comparison for this run.
#
# Target: b200-2 (.144) — idle per 2026-04-26 audit.
# Est. wall time: ~12 min for 2 runs × 16 streams × 32k on 8× L20A.
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
COMMON="--stream_length 32768 --num_streams 16 --skip_chunks 200 \
        --filter_rank 1 --calibration_chunks 64 --sub_window_len 1024 \
        --kv_budget 512 --recent_window 64 \
        --bucket_tokens 2048 --warmup_tokens 4096 \
        --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama3_streaming_32k_${TS}.log
mkdir -p logs

echo "=== Llama-3 streaming 32k rank=1 kv=512/r64 starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local mode="$2"
  local out="outputs/streaming_llama3_32k/${tag}"
  local log="logs/llama3_streaming_32k_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] mode=${mode} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29561 \
    scripts/eval_qfilters_streaming.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --mode "$mode" \
    --filters_cache "$BASEFILTERS" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl ppl_raw tps
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  ppl_raw=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl_raw'])" 2>/dev/null || echo "NA")
  tps=$(python -c "import json;print(f\"{json.load(open('${out}/eval_results.json'))['tokens_per_sec']:.1f}\")" 2>/dev/null || echo "NA")
  echo "[${tag}] mode=${mode} stream=32k×16 rank=1 kv=512 recent=64 -> PPL=${ppl} (raw=${ppl_raw}, ${tps} tok/s)" | tee -a "$ROOT_LOG"
}

# --- Run qfilters and sliding_window back-to-back ---
run_one qf_stream32k_r1_b512   qfilters
run_one sw_stream32k_b512      sliding_window

echo "=== Llama-3 streaming 32k DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/streaming_llama3_32k/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):30s} mode={r['mode']:16s} rank={r['filter_rank']} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} stream={r['stream_length']}×{r['num_streams']:2d} PPL={r['ppl']:.4f} (raw={r['ppl_raw']:.4f}, {r['tokens_per_sec']:.1f} tok/s)\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
