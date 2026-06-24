#!/usr/bin/env bash
# learnmassbias s250 LongEval + RULER W0 eval on diskB node (8 GPUs, shared pool).
# learnmass = winning monotone-SWA arm (W0<W1=36<W2=43<W4=60<...); compare LongEval/RULER vs
# nctx63 SOTA + Landmark baselines (LongEval all-0%; RULER niah_single_1 4/11/14/2).
# W0 closed-book: --chunk_size 512 --swa_eval_chunks 0
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/third_party/babilong-pkg:$ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/outputs/distill_pg19_nctx63_learnmassbias/mem_space_adapter_step000250.pt"
ACFG="$ROOT/outputs/distill_pg19_nctx63_learnmassbias/adapter_config.json"
MODEL="$ROOT/models/Meta-Llama-3-8B"
mkdir -p logs

POOL=$(mktemp); LOCK=$(mktemp)
for L in 4k 8k 16k 32k; do echo "longeval - $L" >> "$POOL"; done
for t in niah_single_1 niah_single_2 niah_multikey_1 variable_tracking; do
  for L in 4k 8k 16k 32k; do echo "ruler $t $L" >> "$POOL"; done
done

pop() { exec 9>"$LOCK"; flock 9; local line; line=$(head -n1 "$POOL"); [ -n "$line" ] && sed -i '1d' "$POOL"; flock -u 9; printf '%s' "$line"; }

worker() {
  local gpu=$1 cell kind t L
  while true; do
    cell=$(pop); [ -z "$cell" ] && break
    read -r kind t L <<< "$cell"
    if [ "$kind" = "longeval" ]; then
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_longeval_mem_space.py \
        --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
        --output_name longeval_learnmass_s250 --lengths "$L" --num_samples 100 \
        --chunk_size 512 --swa_eval_chunks 0 \
        >>logs/eval_learnmass_longeval_ruler_gpu${gpu}.log 2>&1
    else
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_ruler_mem_space.py --model_type mem_space \
        --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
        --output_name ruler_learnmass_s250 --chunk_size 512 --swa_eval_chunks 0 \
        --tasks "$t" --lengths "$L" --num_samples 100 \
        >>logs/eval_learnmass_longeval_ruler_gpu${gpu}.log 2>&1
    fi
  done
}

for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait
echo "LEARNMASS_LONGEVAL_RULER_DONE"
