#!/usr/bin/env bash
# Llama-3.0-8B PPL sweep on WikiText (not pg19), to test whether Q-Filters
# compression behavior is pg19-specific or holds across the "normal"
# long-context benchmark.
#
# Data: data/wikitext_chunks_llama3_4096.npy (5087 chunks × 4096 tokens,
#       uint32, tokenized with models/Llama--Llama3-8b-tokenizer + --no_line_eos)
#
# Sweep mirrors the pg19 post-fix frontier (sw + qf at matched budgets):
#   Dense 4096 (sw with kv=4096 is equivalent to no compression given
#               sub_window_len=1024 and 4 sub-windows, keeping 4096 keys)
#   SWA:  kv ∈ {64, 128, 256, 512}, recent=64
#   QF:   kv ∈ {64, 128, 256, 512}, recent=64 at rank=2
# Separate calibration on WikiText head (first 64 chunks).
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/wikitext_chunks_llama3_4096.npy
TS=$(date +%Y%m%d_%H%M%S)
# WikiText has 5087 chunks; skip_chunks=64 so calibration head does not overlap eval.
COMMON="--seq_length 4096 --skip_chunks 64 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --recent_window 64"
ROOT_LOG=logs/llama3_wiki_sweep_${TS}.log
OUTROOT=outputs/postfix_llama3_wiki
mkdir -p "$OUTROOT"
echo "=== Llama-3 WikiText sweep starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local mode="$2"; local kv="$3"
  local out="${OUTROOT}/${tag}"
  local log="logs/llama3_wiki_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] mode=${mode} kv=${kv} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29537 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --mode "$mode" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] mode=${mode} kv=${kv} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# Dense control: sw mode with kv=4096 (keeps full context)
run_one dense_4096      sliding_window 4096

# SWA sweep
run_one sw_b64_r64      sliding_window  64
run_one sw_b128_r64     sliding_window 128
run_one sw_b256_r64     sliding_window 256
run_one sw_b512_r64     sliding_window 512

# QF sweep (fresh calibration for WikiText head)
run_one qf_b64_r64      qfilters        64
run_one qf_b128_r64     qfilters       128
run_one qf_b256_r64     qfilters       256
run_one qf_b512_r64     qfilters       512

echo "=== Llama-3 WikiText sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Summary ===" | tee -a "$ROOT_LOG"
for d in ${OUTROOT}/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} mode={r['mode']:15s} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
