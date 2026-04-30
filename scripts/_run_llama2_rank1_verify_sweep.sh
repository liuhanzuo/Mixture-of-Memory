#!/usr/bin/env bash
# Llama-2-7B Q-Filters rank=1 universality verification sweep.
#
# Motivation: the Llama-3-8B rank=1 kv-extension sweep (2026-04-26 13:45) found
# a clean monotone-descending PPL curve from kv=96 (PPL=6.92) through kv=1024
# (PPL=2.36). The Llama-2-7B rank=2 curve (§11.4.2, 2026-04-26 13:03) has a
# bowl at kv=96 (PPL=167.27) that rises on both sides. Two hypotheses:
#   H1 (rank effect): at rank=1, Llama-2 also becomes monotone → bowl is
#                     rank-2-specific.
#   H2 (family effect): Llama-2's bowl is intrinsic; rank=1 preserves it.
#
# This sweep runs rank=1 on Llama-2-7B at kv∈{96, 128, 192, 256} to
# discriminate H1 vs H2 directly at shared kv points.
#
# All other hyperparameters IDENTICAL to _run_llama2_kv_lowrange_sweep.sh
# except filter_rank=1. calibration cache is not reused (no pre-existing rank=1
# cache for Llama-2) — each run does its own 64-chunk fresh calibration.
#
# Target: b200-1 (idle 8/8 per 13:50 audit). Est. 4 runs × ~4.5 min ≈ 18 min.
#
# CANONICAL WORKING DIRECTORY: /apdcephfs_wzc1/share_303098609/pighzliu_code
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=models/Llama--Llama2-7b
DATA=data/pg19_chunks.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 1 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama2_rank1_verify_${TS}.log
mkdir -p logs
OUTROOT="${RANK1_VERIFY_OUTDIR:-outputs/rank1_verify_llama2}"
echo "=== Llama-2 Q-Filters rank=1 verification starting $(date) === (OUTROOT=${OUTROOT})" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local kv="$2"; local rw="$3"
  local out="${OUTROOT}/${tag}"
  local log="logs/llama2_rank1_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] rank=1 kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29545 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode qfilters \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] rank=1 kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- rank=1 grid @ shared kv points with Llama-3 rank=1 sweep ---
run_one qf_r1_b96_rw64_llama2  96  64
run_one qf_r1_b128_rw64_llama2 128 64
run_one qf_r1_b192_rw64_llama2 192 64
run_one qf_r1_b256_rw64_llama2 256 64

echo "=== Llama-2 rank=1 verification DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in "${OUTROOT}"/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):28s} mode={r['mode']:10s} rank={r['filter_rank']} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
