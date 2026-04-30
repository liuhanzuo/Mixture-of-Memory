#!/usr/bin/env bash
# Llama-3.0-8B filter_rank sweep at the post-fix 256/64 headline.
# Disentangles the two candidate mechanisms at the post-Patch-A 256/64 headline
# (per ops/research_notes/20260426_s11_retraction.md §11.4 forward plan):
#   (a) GQA 32:8 averaging defeats rank-2 filter subspace
#   (b) Llama-3 sharp-loss regime amplifies compression perturbation
# If PPL drops monotonically with rank, (a) dominates; if flat, (b) dominates.
#
# Sweep: rank in {2, 4, 8}. Each rank requires its own SVD-truncation
# calibration -> new filters.pt. Total ~45 min on b200-1 (8x L20A),
# 3 ranks x ~15 min each.
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --kv_budget 256 --recent_window 64 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --mode qfilters"
ROOT_LOG=logs/llama3_rank_sweep_${TS}.log
echo "=== Llama-3 rank sweep starting $(date) ===" | tee "$ROOT_LOG"

run_rank() {
  local rank="$1"
  local tag="qf_b256_r64_rank${rank}"
  local out="outputs/postfix_llama3_ranksweep/${tag}"
  local log="logs/llama3_rank${rank}_${TS}.log"
  mkdir -p "$out"
  # NOTE: different filter_rank -> different SVD truncation -> each rank needs
  # its own filters.pt. Omit --filters_cache so eval_qfilters.py defaults to
  # "${out}/filters.pt" and runs a fresh calibration on 64 chunks (the same
  # calibration set used for rank=2 / 256/64 / Llama-3-bestpoint).
  echo "=== [rank=${rank}] at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29534 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --filter_rank "$rank" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[rank=${rank}] PPL=${ppl}" | tee -a "$ROOT_LOG"
}

run_rank 2
run_rank 4
run_rank 8

echo "=== Llama-3 rank sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Summary ===" | tee -a "$ROOT_LOG"
for d in outputs/postfix_llama3_ranksweep/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):30s} rank={r['filter_rank']:2d} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
