#!/usr/bin/env bash
# Llama-3.0-8B 2-D (filter_rank x kv_budget) sweep — Patch-A followup.
#
# Scientific purpose:
#   Tests the unified spectral/score-cutoff regularization hypothesis — does
#   the Llama-3 rank-optimum migrate diagonally as kv_budget shrinks, or are
#   rank-optimum and kv-optimum independent?
#   See ops/research_notes/20260426_s11_retraction.md §11.4.2 fold for the
#   motivating analysis (the 1-D rank sweep at kv=256 flagged a possible
#   coupling between spectral truncation and KV score-cutoff).
#
# Grid (row-major over rank so we can bail mid-sweep and still have a rank slice):
#   rank in {1, 2, 4}  x  kv_budget in {128, 256, 512}   -> 9 runs
#   recent_window fixed at 64, calibration_chunks fixed at 64.
#
# Wall-clock estimate:
#   ~30 min on b200-1 (8x L20A), 9 runs x ~3.3 min each (calibration +
#   200-chunk eval). Note filter_rank=1 has not been exercised before.

set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy

# Sanity check: bail early if the tokenized corpus is missing so we don't
# burn 9 torchrun launches on a guaranteed-failing run.
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: data file '$DATA' not found (cwd=$(pwd))." >&2
  echo "  Expected: pre-tokenized Llama-3 PG19 chunks, no-EOS variant." >&2
  echo "  Hint: check scripts/convert_jsonl_to_npy.py or rerun tokenization." >&2
  exit 2
fi

TS=$(date +%Y%m%d_%H%M%S)
# kv_budget is per-call now; leave it out of COMMON.
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --recent_window 64 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --mode qfilters"
ROOT_LOG=logs/llama3_rank_kv_2d_sweep_${TS}.log
echo "=== Llama-3 rank x kv 2-D sweep starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local rank="$1"
  local kv="$2"
  local tag="qf_b${kv}_r64_rank${rank}"
  local out="outputs/patchA_llama3_rank_kv_2d/${tag}"
  local log="logs/llama3_rank${rank}_kv${kv}_${TS}.log"
  mkdir -p "$out"
  # Each (rank, kv) pair gets its own SVD-truncation filters.pt — we omit
  # --filters_cache so eval_qfilters.py defaults to "${out}/filters.pt" and
  # runs a fresh 64-chunk calibration matched to this rank.
  echo "=== [rank=${rank} kv=${kv}] at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29544 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" \
    --filter_rank "$rank" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[rank=${rank} kv=${kv}] PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# Row-major over rank: full kv slice at rank=1 first, then rank=2, then rank=4.
# One call per line (1 def + 9 calls = 10 matching lines in the script).
run_one 1 128
run_one 1 256
run_one 1 512
run_one 2 128
run_one 2 256
run_one 2 512
run_one 4 128
run_one 4 256
run_one 4 512

echo "=== Llama-3 rank x kv 2-D sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Summary ===" | tee -a "$ROOT_LOG"
for d in outputs/patchA_llama3_rank_kv_2d/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):30s} rank={r['filter_rank']:2d} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
