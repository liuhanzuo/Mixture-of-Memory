#!/usr/bin/env bash
# Open-book diagnostic for LongEval: nctx63 SOTA step250 with a 63-chunk SWA window.
# This tests whether LongEval failure is due to closed-book compression/W0 readout.
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/third_party/babilong-pkg:$ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/outputs/distill_pg19_chunk512_nctx63/mem_space_adapter_step000250.pt"
ACFG="$ROOT/outputs/distill_pg19_chunk512_nctx63/adapter_config.json"
MODEL="$ROOT/models/Meta-Llama-3-8B"
mkdir -p logs
POOL=$(mktemp); LOCK=$(mktemp)
for L in 4k 8k 16k 32k; do echo "$L" >> "$POOL"; done
pop() { exec 9>"$LOCK"; flock 9; local line; line=$(head -n1 "$POOL"); [ -n "$line" ] && sed -i '1d' "$POOL"; flock -u 9; printf '%s' "$line"; }
worker() {
  local gpu=$1 L
  while true; do
    L=$(pop); [ -z "$L" ] && break
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_longeval_mem_space.py \
      --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
      --output_name longeval_nctx63_s250_swa63 --lengths "$L" --num_samples 100 \
      --chunk_size 512 --swa_eval_chunks 63 \
      >>logs/eval_nctx63_longeval_swa63_gpu${gpu}.log 2>&1
  done
}
for g in 0 1 2 3; do worker $g & done
wait
echo "NCTX63_LONGEVAL_SWA63_DONE"
