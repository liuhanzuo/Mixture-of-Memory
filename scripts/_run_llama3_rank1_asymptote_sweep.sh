#!/usr/bin/env bash
# Llama-3-8B Q-Filters rank=1 kv-ASYMPTOTE sweep (kv ∈ {2048, 4096}).
#
# Motivation: the rank=1 kv-extension sweep (2026-04-26 13:45) produced a
# monotone-descending curve through kv=1024 (PPL=2.365) with NO plateau.
# Crude power-law fit PPL(kv) ~ 3.7·(kv/512)^−0.5 predicts PPL~1.67 at kv=2048
# and PPL~1.18 at kv=4096 (if the fit holds). But at seq_length=4096, kv=2048
# is 50% compression and kv=4096 is "no compression" — the asymptote must sit
# somewhere in here. This sweep locates it.
#
# Note: kv=4096 with seq_length=4096 means Q-Filters returns every token
# (T <= budget short-circuit in compress_kv). This serves as the dense-floor
# reference point for Llama-3 on this pg19 corpus.
#
# All other hyperparameters IDENTICAL to _run_llama3_rank1_kv_ext_sweep.sh.
#
# Target: b200-2 (idle 8/8 per 13:50 audit). Est. 2 runs × ~4.5 min ≈ 9 min.
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
ROOT_LOG=logs/llama3_rank1_asymptote_${TS}.log
mkdir -p logs
echo "=== Llama-3 Q-Filters rank=1 ASYMPTOTE starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local kv="$2"; local rw="$3"
  local out="outputs/rank1_asymptote_llama3/${tag}"
  local log="logs/llama3_rank1_asym_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] rank=1 kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29546 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode qfilters \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] rank=1 kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- rank=1 asymptote points @ recent=64 ---
run_one qf_r1_b2048_rw64 2048 64
run_one qf_r1_b4096_rw64 4096 64

echo "=== Llama-3 rank=1 asymptote DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/rank1_asymptote_llama3/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):24s} mode={r['mode']:10s} rank={r['filter_rank']} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
