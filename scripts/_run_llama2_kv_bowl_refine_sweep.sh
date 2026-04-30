#!/usr/bin/env bash
# Llama-2-7B Q-Filters rank=2 kv bowl-refinement sweep (kv ∈ {88, 104}).
#
# Motivation: the kv-lowrange sweep (2026-04-26 13:03) showed PPL at
# kv=80 (233.43), kv=96 (167.27), kv=112 (170.67). Bowl bottom is near kv=96
# but we haven't sampled between 80↔96 or 96↔112. This 2-run sweep pins
# the bottom more tightly.
#
# Expected curvature: if the true minimum is AT kv=96 exactly, PPL at 88 and
# 104 should both be above 167.27. If it's off-center, one of them will dip.
#
# All other hyperparameters IDENTICAL to _run_llama2_kv_lowrange_sweep.sh
# and cache `outputs/qfilters_baseline/filters.pt` is reused (same rank=2
# calibration as the bowl sweep).
#
# Target: b200-3 (idle 8/8 per 13:50 audit). Est. 2 runs × ~50 s ≈ 2 min.
#
# CANONICAL WORKING DIRECTORY: /apdcephfs_wzc1/share_303098609/pighzliu_code
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=models/Llama--Llama2-7b
DATA=data/pg19_chunks.npy
TS=$(date +%Y%m%d_%H%M%S)
BASEFILTERS=outputs/qfilters_baseline/filters.pt
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama2_kv_bowl_refine_${TS}.log
mkdir -p logs
echo "=== Llama-2 Q-Filters kv bowl-refine starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local mode="$2"; local kv="$3"; local rw="$4"
  local out="outputs/kv_bowl_refine_llama2/${tag}"
  local log="logs/llama2_kv_bowl_refine_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] mode=${mode} kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29547 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode "$mode" \
    --filters_cache "$BASEFILTERS" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] mode=${mode} kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- kv bowl-refine grid @ recent=64, rank=2 ---
run_one qf_b88_r64  qfilters 88  64
run_one qf_b104_r64 qfilters 104 64

echo "=== Llama-2 kv bowl-refine DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/kv_bowl_refine_llama2/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} mode={r['mode']:15s} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
