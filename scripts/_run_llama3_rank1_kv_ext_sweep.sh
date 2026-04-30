#!/usr/bin/env bash
# Llama-3-8B Q-Filters rank=1 kv-extension sweep.
#
# Motivation: the rank×kv 2-D sweep (2026-04-26 12:51) revealed rank=1 strictly
# dominates at every sampled kv. With only 3 kv points {128,256,512}, the
# rank=1 kv-curve (6.126 / 4.636 / 3.672) is under-sampled. This extends the
# kv axis densely at the winning rank to map the full curve.
#
# 5 new op-points: rank=1 @ kv ∈ {64, 96, 192, 384, 1024}.
# Covers:
#   - below 128: {64, 96} — does rank=1 still dominate in tight-kv regime?
#   - between existing points: {192, 384} — fill the curve
#   - 2× above current max: {1024} — does PPL keep descending or plateau?
#
# All other hyperparameters IDENTICAL to _run_llama3_rank_kv_2d_sweep.sh
# so the new points drop into the existing rank=1 row seamlessly.
#
# Runs sequentially on 8× L20A. Target: any idle B200 node (b200-1 first pick
# because it ran the original 2-D sweep and has warm cache).
# Reuses outputs/qfilters_baseline_llama3/filters.pt (rank=1 calibration
# present from the 2-D sweep, rank=1 row).
#
# CANONICAL WORKING DIRECTORY: /apdcephfs_wzc1/share_303098609/pighzliu_code
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 1 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama3_rank1_kv_ext_${TS}.log
mkdir -p logs
echo "=== Llama-3 Q-Filters rank=1 kv-extension starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local kv="$2"; local rw="$3"
  local out="outputs/rank1_kv_ext_llama3/${tag}"
  local log="logs/llama3_rank1_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] rank=1 kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29543 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode qfilters \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] rank=1 kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- rank=1 kv grid extension @ recent=64 ---
run_one qf_r1_b64_rw64   64   64
run_one qf_r1_b96_rw64   96   64
run_one qf_r1_b192_rw64  192  64
run_one qf_r1_b384_rw64  384  64
run_one qf_r1_b1024_rw64 1024 64

echo "=== Llama-3 rank=1 kv-extension DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/rank1_kv_ext_llama3/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):24s} mode={r['mode']:10s} rank={r['filter_rank']} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
