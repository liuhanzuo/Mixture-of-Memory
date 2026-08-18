#!/usr/bin/env bash
# NFE sweep for Dream-Coder-Instruct-7B on MBPP+.
# Second-benchmark replication of the surviving HE+ finding "low-NFE Pareto
# belongs to structural runtime". Fixed sampler: T=0.1 top_p=0.95 (paper recipe),
# max_new_tokens=512. Vary --steps ∈ {64, 128, 256, 1024}
# (512 already at runs/dream_coder_instruct_mbppplus_r2 -> .807/.680).
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1

for STEPS in 64 128 256 1024; do
  NAME="dream_instruct_mbppplus_nfe${STEPS}"
  OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then
    echo "[$(date '+%F %T')] $NAME exists -> SKIP"; continue
  fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== $NAME ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/generate_evalplus_dream.py \
        --checkpoint models/Dream-Coder-v0-Instruct-7B \
        --dataset mbpp \
        --data-file data/evalplus/mbpp_plus.jsonl \
        --output-dir "$OUT" \
        --steps $STEPS --max-new-tokens 512 --temperature 0.1 --top-p 0.95 \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
  $PY -m evalplus.evaluate --dataset mbpp --samples "$OUT/solutions.jsonl" \
      > "$OUT/evalplus.out" 2>&1
  echo "--- $NAME pass@1 ---"
  grep -E "pass@1|mbpp" "$OUT/evalplus.out" | grep -v "Traceback\|File \"" | tail -4
done
echo "[$(date '+%F %T')] ===== MBPP+ NFE sweep DONE ====="
