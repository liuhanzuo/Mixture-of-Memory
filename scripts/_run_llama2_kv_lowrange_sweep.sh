#!/usr/bin/env bash
# Llama-2-7B Q-Filters kv-LOW-RANGE sweep (kv < 128) to verify §11.4.2 bowl
# hypothesis rejection. Prior fine-sweep {144..240} confirmed monotone
# INCREASING PPL (190.99 @ kv=128 → 278.89 @ kv=256); the true minimum must
# therefore lie at kv ≤ 128.
#
# 4 new op-points: kv_budget ∈ {80,96,112,120} @ recent=64.
# kv=64 intentionally EXCLUDED — known degenerate (recent_window=64 means
# kv_budget=64 leaves 0 slots for non-recent tokens, trivially equal to sink
# degrade case).
#
# All other hyperparameters IDENTICAL to _run_llama2_kv_fine_sweep.sh +
# _run_llama2_sweep_postfix.sh so that the resulting curve extends directly
# beneath the 128 anchor (PPL=190.99).
#
# Runs sequentially on 8× L20A (b200-3 per 2026-04-26 post-migration dispatch).
# Reuses outputs/qfilters_baseline/filters.pt (rank-2, 64 calib chunks;
# calibration is kv-budget-invariant so no recompute).
#
# CANONICAL WORKING DIRECTORY: /apdcephfs_wzc1/share_303098609/pighzliu_code
# (shared across all B200 nodes per user correction 2026-04-26 12:45).
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=models/Llama--Llama2-7b
DATA=data/pg19_chunks.npy
TS=$(date +%Y%m%d_%H%M%S)
BASEFILTERS=outputs/qfilters_baseline/filters.pt
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama2_kv_lowrange_sweep_${TS}.log
mkdir -p logs
echo "=== Llama-2 Q-Filters kv-LOW-RANGE sweep starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local mode="$2"; local kv="$3"; local rw="$4"
  local out="outputs/kv_lowrange_llama2/${tag}"
  local log="logs/llama2_kv_lowrange_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] mode=${mode} kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29542 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode "$mode" \
    --filters_cache "$BASEFILTERS" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] mode=${mode} kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- kv low-range grid (< 128) @ recent=64, rank=2 ---
run_one qf_b80_r64  qfilters 80  64
run_one qf_b96_r64  qfilters 96  64
run_one qf_b112_r64 qfilters 112 64
run_one qf_b120_r64 qfilters 120 64

echo "=== Llama-2 kv-LOW-RANGE sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/kv_lowrange_llama2/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} mode={r['mode']:15s} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
