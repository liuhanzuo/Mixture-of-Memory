#!/usr/bin/env bash
# Llama-3.0-8B kv_budget curve @ filter_rank=2.
# Motivated by the rank sweep result (ops/research_notes/20260425_qfilters_postfix_sweep_analysis.md §8
# + rank sweep addendum): rank=2 dominates rank=4/8 (74.9 < 108/106 PPL), so
# the 48.4x cross-family compression cost is NOT a GQA-averaging problem —
# candidate mechanism is Llama-3's sharp-loss regime. This sweep finds
# whether *any* Llama-3 op-point is competitive.
#
# Reuses: outputs/qfilters_llama3_full_bestpoint/filters.pt (rank=2, 64 calib chunks)
# Sweep:  kv_budget in {64, 128, 512} @ recent=64 (256 already = 74.93)
#         + aggressive-compression op-points mirroring Llama-2 winners:
#           kv=64 recent=16 (Llama-2 best compressed)
#           kv=128 recent=32 (midpoint)
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
BASEFILTERS=outputs/qfilters_llama3_full_bestpoint/filters.pt
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama3_kv_curve_${TS}.log
echo "=== Llama-3 kv_budget curve @ rank=2 starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local mode="$2"; local kv="$3"; local rw="$4"
  local out="outputs/postfix_llama3_kvcurve/${tag}"
  local log="logs/llama3_kvcurve_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] mode=${mode} kv=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29535 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode "$mode" \
    --filters_cache "$BASEFILTERS" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] mode=${mode} kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- kv_budget sweep @ recent=64 ---
run_one qf_b64_r64             qfilters        64  64   # keep_old=0 -> filter-OFF control on Llama-3
run_one qf_b128_r64            qfilters       128  64
run_one qf_b512_r64            qfilters       512  64
# (kv=256 r64 already done in postfix_llama3/qf_b256_r64 = 74.93 PPL)

# --- Aggressive compression tune (mirror Llama-2 winners) ---
run_one qf_b64_r16             qfilters        64  16   # Llama-2 best compressed analog
run_one qf_b128_r32            qfilters       128  32

# --- Sliding-window control at same high-compression point ---
run_one sw_b256_r64            sliding_window 256  64

echo "=== Llama-3 kv_budget curve DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Summary ===" | tee -a "$ROOT_LOG"
for d in outputs/postfix_llama3_kvcurve/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} mode={r['mode']:15s} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
