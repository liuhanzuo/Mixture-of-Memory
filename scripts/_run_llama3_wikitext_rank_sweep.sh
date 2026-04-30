#!/usr/bin/env bash
# Llama-3-8B Q-Filters rank-sweep on WikiText (§11.4 retraction checklist item).
#
# Motivation: the Llama-3 pg19 sweeps (2026-04-26) established that the rank=1
# kv-extension curve descends monotonically with no bowl and no plateau through
# kv=1024, with the dense floor at kv=4096 reaching PPL=1.5468. §11.4 of the
# retraction checklist requires a CROSS-CORPUS verification that the rank=1
# monotone descent is not pg19-specific. This sweep sweeps rank ∈ {1, 2, 4, 8}
# at a fixed mid-curve kv=512 (where rank effects are cleanest on pg19) to
# confirm that on WikiText the same rank ordering holds.
#
# Data: data/wikitext_chunks_llama3_4096.npy  (5087 chunks × 4096 tokens,
#       uint32, tokenized with Llama-3 tokenizer, no line EOS). This is the
#       genuine WikiText-Llama3 npy already used by _run_llama3_wiki_sweep.sh;
#       note the canonical filename differs from the speculative
#       wikitext2_chunks_llama3_noeos.npy referenced in the retraction plan
#       — same corpus, existing tokenization, no pg19 fallback required.
#
# Sweep: rank ∈ {1, 2, 4, 8} at kv_budget=512, recent_window=64. Each run does
#        its OWN 64-chunk fresh calibration on the WikiText head (no shared
#        filters_cache), matching the discipline of
#        _run_llama2_rank1_verify_sweep.sh. All other hyperparameters are
#        IDENTICAL to _run_llama3_rank1_asymptote_sweep.sh.
#
# Target: b200-2 (.144) — idle 8/8 verified at 2026-04-26 launch time.
#         Est. 4 runs × ~4.5 min ≈ 18–20 min.
#
# CANONICAL WORKING DIRECTORY: /apdcephfs_wzc1/share_303098609/pighzliu_code
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/wikitext_chunks_llama3_4096.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama3_wikitext_rank_${TS}.log
mkdir -p logs
echo "=== Llama-3 Q-Filters WikiText rank-sweep starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local rank="$2"; local kv="$3"; local rw="$4"
  local out="outputs/wikitext_rank_sweep_llama3/${tag}"
  local log="logs/llama3_wikitext_rank_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] rank=${rank} kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29548 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --filter_rank "$rank" --kv_budget "$kv" --recent_window "$rw" --mode qfilters \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] rank=${rank} kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- rank sweep at kv=512, recent=64 on WikiText (fresh calibration per rank) ---
run_one qf_r1_b512_rw64 1 512 64
run_one qf_r2_b512_rw64 2 512 64
run_one qf_r4_b512_rw64 4 512 64
run_one qf_r8_b512_rw64 8 512 64

echo "=== Llama-3 WikiText rank-sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/wikitext_rank_sweep_llama3/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):24s} mode={r['mode']:10s} rank={r['filter_rank']} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
