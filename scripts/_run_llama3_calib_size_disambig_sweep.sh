#!/usr/bin/env bash
# Llama-3-8B Q-Filters rank×calibration-size disambiguation sweep.
#
# Motivation: the WikiText rank sweep (2026-04-26, §11.4 retraction checklist)
# exposed a severe rank-degradation curve at fixed calib=64:
#     rank=1 -> PPL 8.57 ; rank=2 -> 11.8x ; rank=4 -> 33.7x ; rank=8 -> 89.38
# Two competing explanations survive the retraction closure:
#     H_A ("calibration-starved"): the SVD truncation at rank >= 4 simply
#         demands more calibration samples before the per-head subspace
#         stabilises; PPL at rank=4/8 should drop sharply as calibration_chunks
#         grows to 256.
#     H_B ("intrinsic rank-regime"): rank >= 4 destroys the filter subspace
#         geometry irrespective of calibration size; PPL curves flatten.
#
# Grid (10 runs): calibration_chunks in {16, 32, 64, 128, 256}
#                 x filter_rank in {4, 8}.
# Fixed:    seq_length=4096, num_chunks=200 (eval), kv_budget=512,
#           recent_window=64, sub_window_len=1024, bf16, sdpa, pg19-noeos
#           (same tokenization used by the rank_sweep reference driver).
#
# Distinguishing signal: if PPL(r=4, c=256) is < 0.8 * PPL(r=4, c=64),
# hypothesis H_A gains support. If the curve flattens (< 20% descent),
# hypothesis H_B wins.
#
# Target: b200-4 (28.89.19.134) - verified idle 4 MiB at launch.
# Est. wall: ~80-120 min; 10 runs x 8-12 min each on 8x L20A; calibration
# dominates wall-time at c=256.
#
# CANONICAL WORKING DIRECTORY: /apdcephfs_wzc1/share_303098609/pighzliu_code
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --kv_budget 512 --recent_window 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --mode qfilters"
ROOT_LOG=logs/llama3_calib_size_disambig_${TS}.log
mkdir -p logs outputs/calib_size_disambig_llama3
echo "=== Llama-3 calib-size x rank disambig sweep starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local rank="$1"; local calib="$2"
  local tag="qf_r${rank}_c${calib}"
  local out="outputs/calib_size_disambig_llama3/${tag}"
  local log="logs/llama3_calib_disambig_${tag}_${TS}.log"
  mkdir -p "$out"
  local t0=$(date +%s)
  echo "=== [${tag}] rank=${rank} calib=${calib} at $(date) ===" | tee -a "$ROOT_LOG"
  # Each (rank, calib) pair requires its own SVD truncation -> fresh filters.pt.
  # Omit --filters_cache so eval_qfilters.py defaults to "${out}/filters.pt" and
  # re-calibrates on --calibration_chunks chunks of pg19 head.
  torchrun --nproc_per_node=8 --master_port=29552 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --filter_rank "$rank" --calibration_chunks "$calib" \
    --output_dir "$out" >> "$log" 2>&1
  local t1=$(date +%s)
  local dt=$((t1 - t0))
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] rank=${rank} calib=${calib} -> PPL=${ppl}  wall=${dt}s" | tee -a "$ROOT_LOG"
}

# --- 10-run grid: {16,32,64,128,256} x {4,8} ---
for calib in 16 32 64 128 256; do
  for rank in 4 8; do
    run_one "$rank" "$calib"
  done
done

echo "=== Llama-3 calib-size disambig sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result table ===" | tee -a "$ROOT_LOG"
for d in outputs/calib_size_disambig_llama3/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} rank={r['filter_rank']:2d} calib={r['calibration_chunks']:4d} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
