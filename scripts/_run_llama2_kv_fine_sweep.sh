#!/usr/bin/env bash
# Llama-2-7B Q-Filters kv-fine-sweep between 128 and 256 to localize the
# Patch-A bowl minimum found by scripts/_run_llama2_sweep_postfix.sh on
# 2026-04-26 (kv=128 PPL=190.99; kv=256 PPL=279; intermediate unsampled).
#
# 7 new op-points: kv_budget ∈ {144,160,176,192,208,224,240} @ recent=64.
# All other hyperparameters IDENTICAL to _run_llama2_sweep_postfix.sh so that
# the resulting curve is directly comparable (same dense/kv128/kv256 anchors).
#
# Runs sequentially on 8× L20A (b200-3 per 2026-04-26 12:27 dispatch).
# Reuses outputs/qfilters_baseline/filters.pt (rank-2, 64 calib chunks;
# calibration is kv-budget-invariant so no recompute).
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=models/Llama--Llama2-7b
DATA=data/pg19_chunks.npy
TS=$(date +%Y%m%d_%H%M%S)
BASEFILTERS=outputs/qfilters_baseline/filters.pt
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama2_kv_fine_sweep_${TS}.log
echo "=== Llama-2 Q-Filters kv-FINE sweep starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local mode="$2"; local kv="$3"; local rw="$4"
  local out="outputs/kv_fine_llama2/${tag}"
  local log="logs/llama2_kv_fine_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] mode=${mode} kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29541 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode "$mode" \
    --filters_cache "$BASEFILTERS" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] mode=${mode} kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- kv fine grid between 128 and 256 @ recent=64 ---
run_one qf_b144_r64 qfilters 144 64
run_one qf_b160_r64 qfilters 160 64
run_one qf_b176_r64 qfilters 176 64
run_one qf_b192_r64 qfilters 192 64
run_one qf_b208_r64 qfilters 208 64
run_one qf_b224_r64 qfilters 224 64
run_one qf_b240_r64 qfilters 240 64

echo "=== Llama-2 kv-FINE sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/kv_fine_llama2/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} mode={r['mode']:15s} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
